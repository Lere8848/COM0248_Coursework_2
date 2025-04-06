import os
import json
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Add parent directory to system path for importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATASET_REALSENSE, get_data, get_num_images

def create_label_json(dataset_path):
    """Create a label JSON file for the specified dataset"""
    # Check dataset path
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist {dataset_path}")
        return False
    
    # Get dataset name
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    print(f"Starting labeling for dataset: {dataset_name}")
    
    # Initialize label dictionary
    label_dict = {}
    
    # Get the number of images
    num_images = get_num_images(dataset_path)
    print(f"Found {num_images} images")
    
    # Process each image
    for idx in range(num_images):
        # Load image
        rgb, depth, _ = get_data(dataset_path, idx)
        if rgb is None:
            print(f"Warning: Unable to load image {idx}")
            continue
        
        # Get image name
        img_name = os.listdir(os.path.join(dataset_path, "image"))[idx]
        frame_name = os.path.splitext(img_name)[0]
        
        # Display RGB image
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.axis('off')
        plt.title(f"image {idx+1}/{num_images}: {frame_name}")
        plt.tight_layout()
        plt.show(block=False)
        
        # Prompt user for input
        valid_input = False
        while not valid_input:
            user_input = input(f"image {idx+1}/{num_images} - {frame_name} has table? (y/n/q): ").lower()
            
            if user_input == 'q':
                plt.close()
                print("User terminated the labeling process")
                # Save labeled results
                if label_dict:
                    save_labels(dataset_path, label_dict)
                return False
            elif user_input == 'y':
                label_dict[frame_name] = 1  # Has table
                valid_input = True
            elif user_input == 'n':
                label_dict[frame_name] = None  # No table
                valid_input = True
            else:
                print("Invalid input, please enter 'y', 'n', or 'q'")
        
        plt.close()
    
    # Save labels to JSON file
    save_labels(dataset_path, label_dict)
    print(f"Dataset {dataset_name} labeling completed!")
    return True

def save_labels(dataset_path, label_dict):
    """Save the label dictionary to a JSON file"""
    # Save path
    label_json_path = os.path.join(dataset_path, "label.json")
    
    # Save labels
    with open(label_json_path, 'w') as f:
        json.dump(label_dict, f, indent=2)
    
    print(f"Labels saved to {label_json_path}")
    print(f"Labeled {len(label_dict)} images")

def extract_folder_key(path):
    """Extract the folder name of the dataset as a key"""
    return os.path.basename(os.path.normpath(path))

def merge_realsense_labels():
    """Merge label files from all RealSense datasets"""
    print("Starting to merge RealSense dataset labels...")
    
    all_labels = {}  # Dictionary for all labels
    folder_labels = {}  # Dictionary for labels organized by folder
    
    # Process each dataset
    for dataset_path in DATASET_REALSENSE:
        folder_key = extract_folder_key(dataset_path)
        label_json_path = os.path.join(dataset_path, "label.json")
        
        print(f"Processing dataset: {folder_key}")
        folder_labels[folder_key] = {}
        
        # Check if label file exists
        if not os.path.exists(label_json_path):
            print(f"  Warning: Label file not found {label_json_path}")
            continue
            
        try:
            # Load label file
            with open(label_json_path, 'r') as f:
                label_dict = json.load(f)
                
            # Count images with/without tables
            has_table = sum(1 for v in label_dict.values() if v is not None)
            no_table = sum(1 for v in label_dict.values() if v is None)
            print(f"  Found {len(label_dict)} labeled images (with table: {has_table}, without table: {no_table})")
            
            # Add labels to the merged dictionary, using "folder_name_frame_name" as the key to avoid conflicts
            for frame_name, label in label_dict.items():
                # Use frame name directly as the key
                all_labels[frame_name] = label
                folder_labels[folder_key][frame_name] = label
                
        except Exception as e:
            print(f"  Error: Failed to process {label_json_path}: {e}")
    
    # Ensure output directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save merged labels to a single file
    all_labels_path = os.path.join("data", "all_realsense_labels.json")
    with open(all_labels_path, 'w') as f:
        json.dump(all_labels, f, indent=2)
    print(f"\nAll labels saved to: {all_labels_path}")
    print(f"Total {len(all_labels)} labels")
    
    # Save labels organized by folder
    folder_labels_path = os.path.join("data", "realsense_folder_labels.json")
    with open(folder_labels_path, 'w') as f:
        json.dump(folder_labels, f, indent=2)
    print(f"Labels organized by folder saved to: {folder_labels_path}")
    
    return all_labels, folder_labels


def main():
    print("RealSense Dataset Labeling Tool")
    print("-------------------------------")
    
    # Display all available datasets
    for i, path in enumerate(DATASET_REALSENSE):
        dataset_name = os.path.basename(os.path.normpath(path))
        print(f"{i+1}. {dataset_name} ({path})")
    
    # Let the user select a dataset
    print("\nOptions:")
    print("a. Label all datasets")
    print("q. Quit")
    
    choice = input("\nPlease select the dataset number to label, or enter 'a' to label all, 'q' to quit: ")
    
    if choice.lower() == 'q':
        print("Program exited")
        return
    elif choice.lower() == 'a':
        # Label all datasets
        for path in DATASET_REALSENSE:
            # Check if label file already exists
            if os.path.exists(os.path.join(path, "label.json")):
                overwrite = input(f"Dataset {os.path.basename(path)} already has a label file. Overwrite? (y/n): ").lower()
                if overwrite != 'y':
                    print(f"Skipping dataset {os.path.basename(path)}")
                    continue
            
            success = create_label_json(path)
            if not success:
                print("Labeling process interrupted")
                break
    else:
        try:
            index = int(choice) - 1
            if 0 <= index < len(DATASET_REALSENSE):
                create_label_json(DATASET_REALSENSE[index])
            else:
                print("Invalid dataset number")
        except ValueError:
            print("Invalid input")

if __name__ == "__main__":
    merge_realsense_labels()
    main()