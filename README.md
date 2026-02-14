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
🚀 Getting Started

1. Prerequisites
   You need Python 3.8+ installed.

2. Installation
   Clone the repo and install the required libraries:

3. Running the Live Demo (Webcam)
   This is the main presentation mode. It opens your default camera and overlays the risk grid in real-time.

Controls: Press q to quit.

Note: If you are presenting alone, bring your face close to the camera or show a picture of a crowd on your phone to trigger the "Red" alert.

4. Running Static Analysis (Batch Processing)
   To process a folder of test images (e.g., from the ShanghaiTech dataset) and generate heatmaps:

Results will be saved in the final_accurate_results/ folder.

🧪 Documentation of Key Scripts
live_camera_demo.py
Purpose: Real-time inference pipeline using OpenCV.

Input: Webcam feed (Source 0).

Preprocessing: Resizes frames to 512x512 and normalizes RGB values.

Inference: PyTorch model predicts 100 density values per frame.

Visualization: Maps values to colors (Green < Yellow < Orange < Red) and blends them onto the video feed.

generate_accurate_heatmaps.py
Purpose: High-fidelity analysis for reports.

Technique: Uses Tiled Inference. Instead of shrinking a large image (which loses detail), it cuts the image into 512x512 tiles, analyzes each tile separately, and stitches the heatmap back together. This ensures maximum accuracy for large surveillance photos.

📦 Dependencies (requirements.txt)
Create a file named requirements.txt and paste this inside:

🔮 Future Improvements
Multi-Camera Fusion: Stitching heatmaps from multiple angles into a single floor plan.

Alert API: Sending automatic webhooks (SMS/Email) to security teams when a "Red Zone" persists for >10 seconds.

Edge Deployment: Optimizing the model with TensorRT to run on Jetson Nano devices.

🏆 Acknowledgments
Dataset: ShanghaiTech Crowd Counting Dataset

Model Backbone: PyTorch ResNet
