import cv2
from midas_depth_estimator import MiDaSDepthEstimator
import matplotlib.pyplot as plt

# 加载图像
img_path = "data/CW2_dataset/data/harvard_c5/hv_c5_1/image/0000001-000000023574.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 加载模型
estimator = MiDaSDepthEstimator(model_path="src/pipelineB/weights/dpt_large_384.pt")

# 预测深度图
depth = estimator.predict(img)

# 可视化
plt.subplot(1, 2, 1)
plt.title("RGB")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Estimated Depth")
plt.imshow(depth, cmap="inferno")
plt.axis("off")

plt.show()
