import numpy as np
import open3d as o3d


paths = ["data/CW2_dataset/harvard_c5/hv_c5_1/",
         "data/CW2_dataset/harvard_c6/hv_c6_1/",
         "data/CW2_dataset/mit_76_studyroom/76-1studyroom2/",
         "data/CW2_dataset/mit_32_d507/d507_2/",
         "data/CW2_dataset/harvard_c11/hv_c11_2/",
         "data/CW2_dataset/mit_lab_hj/lab_hj_tea_nov_2_2012_scan1_erika/",
         "data/CW2_dataset/mit_76_459/76-459b/"]

def get_intrinsics(dataset_path):
    with open(dataset_path + 'intrinsics.txt', 'r') as f:
        lines = f.readlines()
        intrinsics = np.array([float(x) for x in lines[0].split()])
    return intrinsics