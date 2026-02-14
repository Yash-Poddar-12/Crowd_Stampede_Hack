import os
import shutil
import random
import math

# --- CONFIGURATION ---
# Based on your screenshot, your script is next to the 'ShanghaiTech' folder.
BASE_PATH = 'ShanghaiTech'

# Percentage of data to move to validation (0.1 = 10%)
VAL_SPLIT = 0.1 

# Random seed so the split is the same every time you run it
random.seed(42) 

def create_validation_set(part_name):
    print(f"\n--- Processing {part_name} ---")
    
    # 1. Define paths for existing Train Data
    train_img_dir = os.path.join(BASE_PATH, part_name, 'train_data', 'images')
    train_gt_dir = os.path.join(BASE_PATH, part_name, 'train_data', 'ground-truth')

    # 2. Define paths for NEW Validation Data
    val_root_dir = os.path.join(BASE_PATH, part_name, 'validation_data')
    val_img_dir = os.path.join(val_root_dir, 'images')
    val_gt_dir = os.path.join(val_root_dir, 'ground-truth')

    # Check if source exists
    if not os.path.exists(train_img_dir):
        print(f"Error: Could not find training images at {train_img_dir}")
        return

    # 3. Create the folders (SAFE: Won't crash if they already exist)
    if not os.path.exists(val_img_dir):
        print(f"Creating folder: {val_img_dir}")
        os.makedirs(val_img_dir)
        
    if not os.path.exists(val_gt_dir):
        print(f"Creating folder: {val_gt_dir}")
        os.makedirs(val_gt_dir)

    # Get list of images
    all_images = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    total_files = len(all_images)
    
    # Calculate split
    num_val = math.ceil(total_files * VAL_SPLIT)
    print(f"Total Train files: {total_files}")
    print(f"Moving {num_val} files to Validation...")

    # Shuffle and select
    random.shuffle(all_images)
    files_to_move = all_images[:num_val]

    count_moved = 0
    for img_file in files_to_move:
        # Construct filenames
        gt_file = "GT_" + img_file.replace('.jpg', '.mat')

        # Define full source and destination paths
        src_img = os.path.join(train_img_dir, img_file)
        dst_img = os.path.join(val_img_dir, img_file)
        
        src_gt = os.path.join(train_gt_dir, gt_file)
        dst_gt = os.path.join(val_gt_dir, gt_file)

        # Move files
        try:
            shutil.move(src_img, dst_img)
            
            if os.path.exists(src_gt):
                shutil.move(src_gt, dst_gt)
                count_moved += 1
            else:
                print(f"Warning: GT file missing for {img_file}")
                
        except Exception as e:
            print(f"Error moving {img_file}: {e}")

    print(f"Done! Moved {count_moved} image/GT pairs to: {val_root_dir}")

# --- EXECUTION ---
if __name__ == "__main__":
    if os.path.exists(BASE_PATH):
        create_validation_set('part_A')
        create_validation_set('part_B')
        print("\nSUCCESS: Validation folders created with 'images' and 'ground-truth' inside.")
    else:
        print(f"Error: Could not find folder '{BASE_PATH}'. Make sure this script is in the SIMULATOR folder.")