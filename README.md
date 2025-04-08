# COMP0248 Coursework 2  

**Group C**  
Yukai Wang, Yiyang Jia, Zewen Qu  
MSc Robotics and Artificial Intelligence,  
Department of Computer Science,  
University College London

**Contact Emails**:  
yukai.wang.24@ucl.ac.uk  
yiyang.jia.24@ucl.ac.uk  
zewen.qu.24@ucl.ac.uk

---

## Project Introduction

This project implements three different pipelines for **binary classification** and **point cloud segmentation** on a dataset derived from **Sun3D**. The task is to determine whether a **table** is present in a given scene.  

The three pipelines are:

- **Pipeline A**: Depth → Point Cloud → Point Cloud Classification  
- **Pipeline B**: RGB → Depth Estimation → Depth-based Classification  
- **Pipeline C**: Depth → Point Cloud → Point Cloud Segmentation  

---

## Main Features

### Data Processing

- 补充下polygon到点云GT label的Processing。
- **polygon2json**: Converts polygon annotations to JSON format indicating "table" / "no table" presence. (for classification)
- **extract**: Extracts and aligns RGB and depth images from RealSense `.bag` files.
- **realsense_self_label**: Allows manual annotation and labeling for self-captured RealSense data.
- **check_depth**: Utility for depth visualization and validation (e.g. alignment check, missing values).


### Pipeline A – Depth → Point Cloud → Classification

- Converts depth to point cloud
- Applies a PointNet-based classifier
- Evaluates classification performance (Accuracy, Precision, Recall, F1)
- 补充

### Pipeline B – RGB → Depth → Classification

- Uses a pre-trained monocular depth estimation model (e.g. MiDaS)
- Uses the estimated depth to classify presence of a table
- Evaluates both classification (Accuracy, Precision, Recall, F1) and depth estimation (RMSE, MAE)
- 补充

### Pipeline C – Depth → Point Cloud → Segmentation

- Converts depth to point cloud
- Uses a segmentation model to classify points as table or background
- Evaluates per-point IoU and mIoU for segmentation
- 补充

---

## Model Structure

### Pipeline A
补充

### Pipeline B
补充

### Pipeline C
补充

---

## Results

### Pipeline A – Classification on Raw Point Cloud

补充定量metrics，定性可视化的结果啥的

---

### Pipeline B – Depth Estimation + Classification

**Depth Estimation**

| Metric | Value |
|--------|-------|
| RMSE   |       |
| MAE    |       |

**Classification**

| Metric     | Value |
|------------|-------|
| Accuracy   |       |
| Precision  |       |
| Recall     |       |
| F1 Score   |       |

---

### Pipeline C – Point Cloud Segmentation

补充定量metrics，定性可视化的结果啥的

---

## Project Structure
The overall structure of the project is shown below：

```
project_root/
├— data/
│   ├— CW2_dataset/                 # Processed Sun3D dataset
│   ├— realsense_testset/         # Final RealSense test set
│   ├— all_polygon_label_dict.json
│   ├— all_realsense_labels.json
│   ├— polygon2label.py           # Converts polygon to binary mask
│   ├— polygon2json.py            # Converts polygon to table/no-table labels
│   ├— extract.py                 # Extract RGBD from .bag file
│   ├— realsense_self_label.py    # Tool for labelling self data
│   └— check_depth.py             # Utility for checking depth images
├— src/
│   ├— PipelineA/                 # Code for Point Cloud Classification
│   ├— pipelineB/                 # Code for RGB → Depth → Classification
│   ├— PipelineC/                 # Code for Point Cloud Segmentation
│   ├— Dataset.py                # Shared Dataset class
│   ├— utils.py                  # Shared utilities (metrics, transforms)
│   └— env_test.py               # Environment testing script
├— results/                      # Output results, logs, visualizations
├— requirements.txt
└— README.md
```

Below are the specific structures of each pipeline:

### PipelineA Structure
补充
```
src/PipelineA/
│── model.py           # Point cloud classification model
│── train.py           # Training script
│── eval.py            # Evaluation script
└— dataloader.py      # Dataset wrapper for point clouds
```

### pipelineB Structure
补充
```
src/pipelineB/
│── depth_estimator.py # Wrapper for MiDaS or other depth model
│── classifier.py      # Classification model on estimated depth
│── train.py
│── eval.py
```

### PipelineC Structure
补充
```
src/PipelineC/
│── segmenter.py       # Point cloud segmentation model
│── train.py
│── eval.py
│── utils.py           # For point cloud coloring and visualization
```

---

## How to Run

### 1. Install Dependencies

Recommended Python version: **这里到时候确认下**  
Install required packages with:

```bash
pip install -r requirements.txt
```

---

### 2. Run Pipeline A

```bash
cd src/PipelineA
补充下
```

---

### 3. Run Pipeline B

```bash
cd src/pipelineB
补充
```

---

### 4. Run Pipeline C

```bash
cd src/PipelineC
补充
```

---

## References
补充下参考的几个模型
- [MiDaS: Accurate Monocular Depth Estimation](https://github.com/isl-org/MiDaS)
- [Sun3D Dataset](https://sun3d.cs.princeton.edu)

---

