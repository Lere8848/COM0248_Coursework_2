import os
import cv2
import numpy as np

# Define the directory containing the PNG images
directory = "data/realsense/innovation_lab_outside/depth_raw"

# Initialize variables to track the min and max values
min_value = float('inf')
max_value = float('-inf')

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        filepath = os.path.join(directory, filename)
        # Read the image as a grayscale image
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is not None:
            # Check the bit depth of the image
            bit_depth = image.dtype
            # Get the image dimensions
            height, width = image.shape
            print(f"File: {filename}, Bit Depth: {bit_depth}, Size: {width}x{height}")
            # Update min and max values
            min_value = min(min_value, np.min(image))
            max_value = max(max_value, np.max(image))

# Output the range of depth values
print(f"Depth PNG value range: Min = {min_value}, Max = {max_value}")