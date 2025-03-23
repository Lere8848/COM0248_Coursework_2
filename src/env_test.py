import torch
import open3d as o3d

def test_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch can use GPU.")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. PyTorch cannot use GPU.")

def test_open3d():
    try:
        print("Testing Open3D...")
        mesh = o3d.geometry.TriangleMesh.create_sphere()
        print("Open3D is working correctly.")
    except Exception as e:
        print(f"Open3D test failed: {e}")

if __name__ == "__main__":
    print("Testing PyTorch with CUDA...")
    test_cuda()
    print("\nTesting Open3D...")
    test_open3d()