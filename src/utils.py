import os
import numpy as np
import open3d as o3d
import pickle
import matplotlib.pyplot as plt


DATASET_PATHS = ["data/CW2_dataset/harvard_c5/hv_c5_1/",
         "data/CW2_dataset/harvard_c6/hv_c6_1/",
         "data/CW2_dataset/mit_76_studyroom/76-1studyroom2/",
         "data/CW2_dataset/mit_32_d507/d507_2/",
         "data/CW2_dataset/harvard_c11/hv_c11_2/",
         "data/CW2_dataset/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/",
         "data/CW2_dataset/mit_76_459/76-459b/"]

NUM_IMAGES = [23, 35, 48, 108, 13, 13, 43]

def check_time(func):
    """
    A decorator that prints the time a function takes
    """
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__} took {time.time() - start} seconds')
        return result
    return wrapper

def debuger(func):
    """
    A decorator that prints the arguments and the return value of a function
    """
    def wrapper(*args, **kwargs):
        print(f'Arguments: {args}, {kwargs}')
        result = func(*args, **kwargs)
        print(f'Return value: {result}')
        return result
    return wrapper

def get_num_images(dataset_path):
    """
    get number of images in the dataset
    """
    return len(os.listdir(dataset_path+"image")) - 1

def get_intrinsics(dataset_path):
    """
    get 3x3 intrinsics matrix from dataset
    """
    with open(dataset_path + 'intrinsics.txt', 'r') as f:
        intrinsics = f.readlines()
    intrinsics = np.array([list(map(float, line.strip().split())) for line in intrinsics])
    return intrinsics

def get_data(dataset_path,data_id):
    """
    get rgb, depth, labels
    """
    # Check if data_id is valid
    if data_id < 0 or data_id > get_num_images(dataset_path):
        return None, None, None
    # ================== Load RGB ==================
    # get name of the i-th image
    img_name = os.listdir(dataset_path+"image")[data_id]
    # load the image
    rgb = plt.imread(dataset_path+"image/"+img_name)

    # ================== Load Depth ==================
    # get name of the i-th depth image
    depth_name = os.listdir(dataset_path+"depthTSDF")[data_id]
    # load the depth image
    depth = plt.imread(dataset_path+"depthTSDF/"+depth_name)

    # ================== Load Labels ==================
    with open(dataset_path+"labels/tabletop_labels.dat", 'rb') as label_file:
        tabletop_labels = pickle.load(label_file)
        label_file.close()
    labels = tabletop_labels[data_id]
    
    return rgb, depth, labels

def visualize_data(rgb, depth, labels):
    """
    visualize rgb, depth, labels
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title('RGB')

    plt.subplot(132)
    plt.imshow(depth, cmap='gray')
    plt.axis('off')
    plt.title('Depth')

    plt.subplot(133)
    plt.imshow(rgb)
    for polygon in labels:
        plt.plot(polygon[0]+polygon[0][0:1],polygon[1]+polygon[1][0:1],'r')
    plt.axis('off')
    plt.title('Labels')

    plt.show()


def depth_to_point_cloud(depth, K):
    """
    depth: (H, W) depth image
    K: (3, 3) intrinsics matrix
    """
    # Get image size and pixel coordinates
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Convert pixel coordinates to normalized camera coordinates
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    z = depth.flatten()
    # Convert normalized camera coordinates to camera coordinates
    x = (x.flatten() * z)
    y = (y.flatten() * z)
    points = np.vstack((x, y, z)).T
    points = points[z > 0]  # remove points with 0 depth
    return points

def visualize_point_cloud(points):
    """
    points: Nx3 point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])




if __name__ == '__main__':
    # Load data
    dataset_path = DATASET_PATHS[0]
    data_id = 0
    rgb, depth, labels = get_data(dataset_path, data_id)
    visualize_data(rgb, depth, labels)

    # Load intrinsics
    intrinsics = get_intrinsics(dataset_path)
    print(intrinsics)

    # Convert depth to point cloud
    points = depth_to_point_cloud(depth, intrinsics)
    visualize_point_cloud(points)