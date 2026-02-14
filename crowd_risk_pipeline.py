import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION ----------------
IMAGE_PATH = r"ShanghaiTech/part_A/test_data/images/IMG_10.jpg"

YOLO_MODEL = "yolov8m.pt"     # good balance speed/accuracy
CONFIDENCE_THRESHOLD = 0.30

DBSCAN_EPS = 60               # distance in pixels (tune later)
DBSCAN_MIN_SAMPLES = 4        # people required to form risky cluster

GRID_SIZE = 10
HEATMAP_SCALE = 4             # 10x10 -> 40x40
# ------------------------------------------------


# ---------------- STEP 1: DETECT PEOPLE ----------------
def detect_people(image):
    import os
    # Force CPU usage for ultralytics/pytorch ops to avoid CUDA-nms mismatch
    # (slower but works immediately)
    # Option A - hide CUDA from PyTorch:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("Loading YOLO model (CPU mode)...")
    model = YOLO(YOLO_MODEL)

    print("Detecting people (device=cpu)...")
    # Explicitly tell model.predict to use CPU
    results = model.predict(image, conf=CONFIDENCE_THRESHOLD, classes=0, verbose=False, device='cpu')

    points = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        points.append([center_x, center_y])

    return np.array(points)


# ---------------- STEP 2: CLUSTER ----------------
def cluster_people(points):
    if len(points) == 0:
        return np.array([]), np.array([])

    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(points)
    labels = clustering.labels_

    risky_points = points[labels != -1]

    return labels, risky_points


# ---------------- STEP 3: CREATE 10x10 GRID ----------------
def create_grid(risky_points, height, width):
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    if len(risky_points) == 0:
        return grid

    cell_w = width / GRID_SIZE
    cell_h = height / GRID_SIZE

    for x, y in risky_points:
        col = int(min(x // cell_w, GRID_SIZE - 1))
        row = int(min(y // cell_h, GRID_SIZE - 1))
        grid[row, col] += 1

    return grid


# ---------------- STEP 4: VISUALIZATION ----------------
def visualize(image, all_points, risky_points, grid):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    heatmap = ndi.zoom(grid, HEATMAP_SCALE, order=3)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ---- Tactical View ----
    axes[0].imshow(rgb_image)
    axes[0].set_title("Tactical View")

    if len(all_points) > 0:
        axes[0].scatter(all_points[:, 0], all_points[:, 1], c='lime', s=8)

    if len(risky_points) > 0:
        axes[0].scatter(risky_points[:, 0], risky_points[:, 1], c='red', s=30, marker='x')

    # Overlay grid heat
    masked_grid = np.ma.masked_where(grid < 1, grid)
    axes[0].imshow(masked_grid, cmap='jet', alpha=0.4,
                   extent=(0, width, height, 0))

    axes[0].axis("off")

    # ---- Strategic Heatmap ----
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("40x40 Heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.show()


# ---------------- MAIN ----------------
def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Image not found.")
        return

    height, width = image.shape[:2]

    # Detect
    all_points = detect_people(image)
    print("Total people detected:", len(all_points))

    # Cluster
    labels, risky_points = cluster_people(all_points)
    print("People in risky clusters:", len(risky_points))

    # Grid
    grid = create_grid(risky_points, height, width)

    # Visualize
    visualize(image, all_points, risky_points, grid)


if __name__ == "__main__":
    main()
