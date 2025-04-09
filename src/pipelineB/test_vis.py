import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import DATASET_REALSENSE, get_data, get_num_images

from depth_estimator_midas import MiDaSDepthEstimator
from classifier_cnn_mlp import CNNMLPDepthClassifier
from pipelineB_model import PipelineBModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "weights/pipelineB"
MODEL_PATH = os.path.join(SAVE_DIR, 'best_pipelineB_model.pth')

def main():
    print("Initializing model...")
    midas = MiDaSDepthEstimator(model_path="src/pipelineB/weights/dpt_hybrid_384.pt", device=DEVICE)
    cnn_mlp = CNNMLPDepthClassifier(num_classes=2).to(DEVICE)
    model = PipelineBModel(midas, cnn_mlp, freeze_midas=True).to(DEVICE)

    print(f"Loading model weights: {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Iterate through all datasets
    for dataset_path in DATASET_REALSENSE:
        print(f"\nProcessing dataset: {dataset_path}")
        
        try:
            # Get the number of images in the dataset
            num_images = get_num_images(dataset_path)
            print(f"Found {num_images} images")
            
            for idx in range(num_images):
                print(f"Processing image {idx+1}/{num_images}")
                
                # Get data for a single image
                rgb, _, _ = get_data(dataset_path, idx)
                
                if rgb is None:
                    print(f"Skipping image {idx+1}: Unable to load data")
                    continue
                
                # Convert to PyTorch tensor
                rgb_tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
                
                # Add batch dimension and move to device
                rgb_batch = rgb_tensor.unsqueeze(0).to(DEVICE)
                
                # Prediction
                with torch.no_grad():
                    outputs = model(rgb_batch)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    prediction = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][prediction].item()
                
                # Create plot
                plt.figure(figsize=(8, 6))
                
                # Display image
                ax = plt.subplot(1, 1, 1)
                ax.imshow(rgb)
                ax.axis('off')
                
                # Add prediction result text (bottom)
                result_text = "has table" if prediction == 1 else "no table"
                confidence_text = f"conf: {confidence:.2f}"
                plt.figtext(0.5, 0.01, f"pred output: {result_text} | {confidence_text}", 
                          ha="center", fontsize=16, 
                          bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
                
                # Add dataset information
                plt.suptitle(f"data: {os.path.basename(dataset_path)} | img {idx+1}/{num_images}", 
                           fontsize=14)
                
                plt.tight_layout()
                plt.show(block=False)
                
                # Wait for user input
                key = input("Press Enter to continue to the next image, type 'q' to quit: ")
                if key.lower() == 'q':
                    return
                
                plt.close()
                
        except Exception as e:
            print(f"Error processing dataset {dataset_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()