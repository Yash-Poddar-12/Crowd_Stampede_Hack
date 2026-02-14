import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to the data we just created
DATASET_DIR = 'final_model_dataset/train' # Check training data first

def verify_random_samples():
    vis_dir = os.path.join(DATASET_DIR, 'visualizations')
    lbl_dir = os.path.join(DATASET_DIR, 'labels')
    
    if not os.path.exists(vis_dir):
        print("Error: Dataset not found. Did you run prepare_dataset.py?")
        return

    # Get all JSON files
    all_files = [f for f in os.listdir(lbl_dir) if f.endswith('.json')]
    
    if not all_files:
        print("No data found!")
        return

    # Pick 3 random files
    samples = random.sample(all_files, 3)
    
    for json_file in samples:
        file_id = json_file.replace('.json', '')
        
        # Load JSON
        with open(os.path.join(lbl_dir, json_file), 'r') as f:
            data = json.load(f)
            
        # Load Image
        img_path = os.path.join(vis_dir, file_id + '.png')
        
        print(f"\n--- Verifying {file_id} ---")
        print(f"Risky Sectors Detected: {list(data['readable_risks'].keys())}")
        print(f"AI Vector Summary (Non-zero counts): {sum(1 for x in data['grid_vector'] if x > 0)}")
        
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"{file_id}\nVector Sum: {sum(data['grid_vector'])}")
            plt.axis('off')
            plt.show()
        else:
            print("Warning: Matching image file not found.")

if __name__ == "__main__":
    verify_random_samples()