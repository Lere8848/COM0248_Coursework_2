import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import open3d as o3d
from src.utils import depth_to_point_cloud, get_data, get_intrinsics,get_num_images, visualize_data
import cv2
from tqdm import tqdm

# path to train set
DATASET_PATHS = [
    "data/CW2_dataset/mit_76_studyroom/76-1studyroom2/",
    "data/CW2_dataset/mit_32_d507/d507_2/",
    "data/CW2_dataset/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/",
    "data/CW2_dataset/mit_76_459/76-459b/",
    "data/CW2_dataset/mit_gym_z_squash/gym_z_squash_scan1_oct_26_2012_erika/",
]
SAVE_DIR = "data/train_data"

# uncomment this to genetrate the data for the test set
# DATASET_PATHS = [
#     "data/CW2_dataset/harvard_c6/hv_c6_1/",
#     "data/CW2_dataset/harvard_c11/hv_c11_2/",
#     "data/CW2_dataset/harvard_c5/hv_c5_1/",
#     "data/CW2_dataset/harvard_tea_2/hv_tea2_2/",
# ]
# SAVE_DIR = "data/test_data"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_POINTS = 4096

def random_sampling(points, num_samples):
    if points.shape[0] > num_samples:
        idx = np.random.choice(points.shape[0], num_samples, replace=False)
        return points[idx]
    return points

def depth_to_pointcloud_with_labels(depth, K, polygons, image_size):
    """
    Args:
        depth: (H, W) numpy array, depth image
        K: (3, 3) camera intrinsics
        polygons: list of list of (x, y), e.g. [[(x1, y1), (x2, y2), ...], ...]
        image_size: (H, W), same as depth

    Returns:
        points: (N, 3) point cloud
        labels: (N,) array of 0 or 1
    """
    # Generate full pixel grid
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_flat = u.flatten()
    v_flat = v.flatten()
    z = depth.flatten()

    # Camera intrinsics
    x = (u_flat - K[0, 2]) / K[0, 0]
    y = (v_flat - K[1, 2]) / K[1, 1]
    x *= z
    y *= z

    points = np.vstack((x, y, z)).T

    # filter out invalid points (z <= 0)
    valid_mask = z > 0
    points = points[valid_mask]
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]

    # Create polygon mask (in image space)
    mask = np.zeros(image_size, dtype=np.uint8)
    for polygon in polygons:
        polygon_np = np.array([polygon], dtype=np.int32)
        cv2.fillPoly(mask, polygon_np, 1)

    # check if the polygon is valid
    labels = mask[v_valid, u_valid]  # v is row, u is column
    return points, labels

def extract_polygons_from_labels(labels):
    """
    Convert labels like [(x_list, y_list), ...] to OpenCV polygon format.
    Returns:
        List of list of (x, y) integer tuples.
    """
    polygons = []
    if labels is not None:
        for (x_list, y_list) in labels:
            polygon = list(zip(map(int, x_list), map(int, y_list)))
            polygons.append(polygon)
    return polygons

def balanced_sample_pointcloud(points, labels, num_points, table_ratio):
    """
    samped points and labels to make sure the table points are taking more points
    Args:
        points: (N, 3) point cloud
        labels: (N,) array of 0 or 1
        num_points: total number of points to sample
        table_ratio: ratio of table points in the sampled points
    """
    table_points = points[labels == 1]
    bg_points = points[labels == 0]

    num_table = int(num_points * table_ratio)
    num_bg = num_points - num_table

    # if the points are less than num_table or num_bg, we will sample with replacement
    def sample_with_repeat(pts, n):
        if len(pts) == 0:
            return np.zeros((n, 3))  # if no points, return zero points
        elif len(pts) >= n:
            idx = np.random.choice(len(pts), n, replace=False)
        else:
            idx = np.concatenate([
                np.arange(len(pts)),
                np.random.choice(len(pts), n - len(pts), replace=True)
            ])
        return pts[idx]

    sampled_table = sample_with_repeat(table_points, num_table)
    sampled_bg = sample_with_repeat(bg_points, num_bg)

    points_sampled = np.concatenate([sampled_table, sampled_bg], axis=0)
    labels_sampled = np.concatenate([
        np.ones(len(sampled_table), dtype=np.int64),
        np.zeros(len(sampled_bg), dtype=np.int64)
    ], axis=0)
    # shuffle the sampled points and labels
    perm = np.random.permutation(num_points)
    return points_sampled[perm], labels_sampled[perm]





def save_all_processed_data():
    for dataset_path in DATASET_PATHS:
        scene_name = dataset_path.strip("/").split("/")[-2]
        num_images = get_num_images(dataset_path)

        save_dir = os.path.join(SAVE_DIR, scene_name)
        os.makedirs(save_dir, exist_ok=True)

        for data_id in tqdm(range(num_images), desc=scene_name):
            rgb, depth, label_polygons = get_data(dataset_path, data_id)
            if depth is None:
                continue

            intrinsics = get_intrinsics(dataset_path)
            img_size = depth.shape[:2]

            # save negative samples if no labels
            if label_polygons is None or len(label_polygons) == 0:
                points_full = depth_to_point_cloud(depth, intrinsics)
                points_sampled = random_sampling(points_full, NUM_POINTS)
                labels_sampled = np.zeros(NUM_POINTS, dtype=np.int64)
                # # visualize the sampled points and labels
                colors = np.zeros_like(points_sampled)
                colors[:] = [0.5, 0.5, 0.5] 
                colors[labels_sampled == 1] = [1.0, 0.0, 0.0] 

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_sampled)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.visualization.draw_geometries([pcd])
                save_path = os.path.join(save_dir, f"{scene_name}_{data_id}.npz")
                np.savez_compressed(save_path,
                                    points=points_sampled.astype(np.float32),
                                    labels=labels_sampled)
                print(f"[{scene_name}_{data_id}] Saved negative sample.")
                continue

            polygons = extract_polygons_from_labels(label_polygons)
            points, point_labels = depth_to_pointcloud_with_labels(depth, intrinsics, polygons, img_size)

            if len(points) < 10:
                continue

            # save negative samples if no table points
            if np.sum(point_labels == 1) == 0:
                points_sampled = random_sampling(points, NUM_POINTS)
                labels_sampled = np.zeros(NUM_POINTS, dtype=np.int64)

                save_path = os.path.join(save_dir, f"{scene_name}_{data_id}.npz")
                np.savez_compressed(save_path,
                                    points=points_sampled.astype(np.float32),
                                    labels=labels_sampled)
                print(f"[{scene_name}_{data_id}] Saved negative sample (no table).")
                continue

            # sample with table points
            points_sampled, labels_sampled = balanced_sample_pointcloud(points, point_labels, NUM_POINTS, table_ratio=0.2)
            # visualize the sampled points and labels
            colors = np.zeros_like(points_sampled)
            colors[:] = [0.5, 0.5, 0.5] 
            colors[labels_sampled == 1] = [1.0, 0.0, 0.0] 

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_sampled)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([pcd])
            print("The table point occupied:", np.sum(labels_sampled) / NUM_POINTS)

            save_path = os.path.join(save_dir, f"{scene_name}_{data_id}.npz")
            np.savez_compressed(save_path,
                                points=points_sampled.astype(np.float32),
                                labels=labels_sampled)

            
if __name__ == "__main__":
    save_all_processed_data()
