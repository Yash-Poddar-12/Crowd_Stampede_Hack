🛡️ Crowd Risk Simulator: Real-Time Stampede Prevention
A computer vision system for proactive crowd density analysis and crush prevention.

🚨 The Problem
Crowd disasters (stampedes and crushes) are often preventable. They occur when crowd density exceeds a critical threshold (approx. 6 people/m²), creating "fluid-like" shockwaves where individuals lose control of their movement. Traditional surveillance relies on human operators who cannot quantify density in real-time across hundreds of cameras.

💡 The Solution
We built a Deep Learning Density Regressor that analyzes video feeds in real-time. Instead of simply counting people (which fails in high density due to occlusion), our system estimates Crowd Pressure.

Tactical View: A 10x10 overlay grid showing immediate "Red Zones" (Critical Density).

Strategic Heatmap: A smooth, weather-radar style map visualization of crowd pressure gradients.

Privacy-First: The system analyzes texture patterns, not faces. No biometric data is stored or identified.

🏗️ Technical Architecture
We utilized a modified ResNet-18 (Residual Neural Network) architecture.

Backbone: ResNet-18 (Pre-trained on ImageNet) extracts complex texture features from the scene.

Modification: We replaced the final Classification Layer (which usually detects objects) with a Regression Head.

Output: A 100-dimensional vector representing a 10x10 Density Grid.

Post-Processing:

Bicubic Interpolation: Smooths the 10x10 grid into a high-resolution heatmap.

Sensitivity Boosting: Applies a multiplier (2.0x) to detect smaller clusters (3-4 people) as potential risk zones in lower-density environments.

📂 Repository Structure

- `create_new_model.py`: Script to create a new model architecture.
- `create_validation.py`: Script for creating validation datasets.
- `crowd_risk_model.pth`: Trained PyTorch model weights for crowd risk estimation.
- `crowd_risk_pipeline.py`: Main pipeline script for crowd risk analysis.
- `dbscan_model.py`: Implementation of DBSCAN clustering for crowd detection.
- `detect_and_cluster.py`: Script to detect and cluster people in images.
- `evaluate_model_v3.py`, `evaluate_model.py`: Scripts for evaluating model performance.
- `generate_accurate_heatmaps.py`: Generates high-fidelity heatmaps using tiled inference.
- `generate_graded_risk_data.py`: Generates graded risk data for training.
- `generate_heatmaps.py`: Basic heatmap generation script.
- `live_camera_demo.py`: Real-time demo using webcam feed.
- `live_crowd_analysis.py`: Live crowd analysis script.
- `live_demo.py`: General live demo script.
- `live_heatmap_demo_batch.py`: Batch processing for live heatmap demos.
- `live_heatmap_demo.py`: Live heatmap demo.
- `live_heatmap_smooth.py`: Smooth heatmap generation for live feeds.
- `live_heatmap_tiled.py`: Tiled heatmap for live analysis.
- `matlab_data.py`: Script for handling MATLAB data formats.
- `prepare_dataset.py`: Prepares datasets for training.
- `README.md`: This file.
- `real_time_pipeline.py`: Real-time processing pipeline.
- `requirements.txt`: Python dependencies.
- `run_batch_yolo.py`: Runs YOLO in batch mode.
- `run_full_test.py`: Full test suite script.
- `train_model_v2.py`, `train_model_v3_balanced.py`, `train_model_v4_robust.py`, `train_model.py`: Training scripts for different model versions.
- `training_model_cluster.py`: Clustering-based training script.
- `verify_data.py`: Data verification script.
- `vizualization.py`: Visualization utilities (note: typo in filename).
- `yolov8m.pt`, `yolov8x.pt`: Pre-trained YOLOv8 model weights.
- `evaluation_results/`: Folder for evaluation outputs.
- `final_accurate_results/`: High-accuracy results from tiled inference.
- `final_heatmap_results/`: Final heatmap visualizations.
- `final_metrics_report/`: Reports on model metrics.
- `final_model_dataset/`: Processed dataset for the final model (with train/val/test splits).
- `final_smoothed_results/`: Smoothed result outputs.
- `heatmap_results/`, `heatmap_results_combined/`, `heatmap_results_smooth/`, `heatmap_results_tiled/`: Various heatmap result folders.
- `results/`: General results folder.
- `ShanghaiTech/`: ShanghaiTech crowd counting dataset (parts A and B).
- `training_data_graded_risk/`: Graded risk training data.
- `training_data_model_data/`: Model-specific training data.
- `yolo_batch_results/`: Batch results from YOLO processing.

🚀 Getting Started

1. Prerequisites
   - Python 3.8 or higher
   - Git (for cloning the repository)
   - Webcam (for live demos, optional)

2. Installation
   Clone the repository and install the required dependencies:

   ```bash
   git clone <repository-url>
   cd simulator
   pip install -r requirements.txt
   ```

   Note: If you encounter issues with PyTorch, install it separately based on your CUDA version:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
   ```

3. Running the Live Demo (Webcam)
   This is the main presentation mode. It opens your default camera and overlays the risk grid in real-time.

   ```bash
   python live_camera_demo.py
   ```

   Controls: Press 'q' to quit.

   Note: If presenting alone, bring your face close to the camera or show a picture of a crowd on your phone to trigger the "Red" alert.

4. Running Static Analysis (Batch Processing)
   To process a folder of test images (e.g., from the ShanghaiTech dataset) and generate heatmaps:

   ```bash
   python generate_accurate_heatmaps.py --input_folder ShanghaiTech/part_A/test_data/images --output_folder final_accurate_results/
   ```

   Results will be saved in the `final_accurate_results/` folder.

5. Training a New Model
   To train the crowd risk model:

   ```bash
   python train_model.py
   ```

   Or use one of the variant scripts like `train_model_v4_robust.py` for different configurations.

6. Evaluating the Model
   To evaluate model performance:

   ```bash
   python evaluate_model.py
   ```

7. Preparing Dataset
   To prepare the dataset for training:

   ```bash
   python prepare_dataset.py
   ```

🧪 Documentation of Key Scripts

- `live_camera_demo.py`: Real-time inference pipeline using OpenCV. Input: Webcam feed. Preprocessing: Resize to 512x512, normalize. Inference: Predict 100 density values. Visualization: Color-coded overlay.
- `generate_accurate_heatmaps.py`: High-fidelity analysis using tiled inference for large images. Cuts images into tiles, processes each, stitches results.
- `train_model.py`: Trains the ResNet-18 based density regressor on crowd data.
- `evaluate_model.py`: Evaluates model accuracy using metrics like MAE, MSE on test data.
- `detect_and_cluster.py`: Uses YOLO for person detection and DBSCAN for clustering.
- `real_time_pipeline.py`: End-to-end real-time processing pipeline.
- `run_batch_yolo.py`: Batch processing with YOLO for crowd detection.
- Other scripts: Various utilities for data preparation, visualization, and testing.

📦 Dependencies (requirements.txt)

```
ultralytics
opencv-python
scikit-learn
matplotlib
scipy
numpy
torch
torchvision
```

Note: PyTorch versions may need to be adjusted based on your system (CPU/GPU, CUDA version).

🔮 Future Improvements

- Multi-Camera Fusion: Stitching heatmaps from multiple angles into a single floor plan.
- Alert API: Sending automatic webhooks (SMS/Email) to security teams when a "Red Zone" persists for >10 seconds.
- Edge Deployment: Optimizing the model with TensorRT to run on Jetson Nano devices.

🏆 Acknowledgments

- Dataset: ShanghaiTech Crowd Counting Dataset
- Model Backbone: PyTorch ResNet-18
