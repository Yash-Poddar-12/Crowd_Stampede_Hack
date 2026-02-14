# detect_and_cluster.py
import os, json
import cv2, numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

# -------- USER CONFIG ----------
IMAGE_PATH = r"ShanghaiTech/part_A/test_data/images/IMG_10.jpg"
YOLO_WEIGHTS = "yolov8m.pt"   # or path to your fine-tuned weights
CONF_THRESH = 0.30
FORCE_CPU = True              # set True if you got torchvision::nms CUDA error
GRID_SIZE = 10
HEAT_SCALE = 4
BASE_MIN_SAMPLES = 4
EPS_SCALE = 1.2               # scale applied to estimated eps
OUT_FOLDER = "results"
# -------------------------------

os.makedirs(OUT_FOLDER, exist_ok=True)

def load_model(weights, device='cpu'):
    return YOLO(weights)

def detect_centers(img, model, conf=0.3, device='cpu'):
    # model.predict accepts numpy image as well; use device param
    results = model.predict(img, conf=conf, classes=0, device=device, verbose=False)
    pts = []
    # results[0].boxes may be empty
    for b in results[0].boxes:
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
        pts.append(((x1+x2)/2.0, (y1+y2)/2.0))
    return np.array(pts)

def auto_dbscan(points, base_min_samples=4, eps_scale=1.2, k_for_eps=4):
    if len(points) == 0:
        return np.array([]), np.array([]), None
    if len(points) < base_min_samples:
        labels = np.array([-1]*len(points))
        return labels, np.array([]), None

    k = min(k_for_eps, len(points)-1)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances, _ = nbrs.kneighbors(points)
    # median of nearest neighbor distances (skip self)
    eps_est = np.median(distances[:, 1:k+1])
    eps = max(3.0, eps_est * eps_scale)
    clustering = DBSCAN(eps=eps, min_samples=base_min_samples).fit(points)
    labels = clustering.labels_
    risky = points[labels != -1]
    return labels, risky, eps

def points_to_grid(points, H, W, rows=10, cols=10):
    grid = np.zeros((rows, cols), dtype=float)
    if len(points)==0:
        return grid
    cell_w = W / cols
    cell_h = H / rows
    for x,y in points:
        c = int(min(x // cell_w, cols-1))
        r = int(min(y // cell_h, rows-1))
        grid[r,c] += 1
    return grid

def density_heatmap(points, H, W, rows=10, cols=10, scale=4, sigma=1.2):
    # grid accumulate then gaussian smooth + zoom
    grid = points_to_grid(points, H, W, rows, cols)
    smooth = ndi.gaussian_filter(grid, sigma=sigma)
    heat = ndi.zoom(smooth, scale, order=3)
    if heat.max()>0:
        heat = heat / heat.max()
    return heat

def visualize_and_save(img, all_points, risky_points, grid, heat_density, out_path):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H,W = img.shape[:2]
    heat_cluster = ndi.zoom(grid, HEAT_SCALE, order=3)
    vmax = max(heat_cluster.max(), heat_density.max(), 1e-6)

    fig, axes = plt.subplots(1,3, figsize=(20,7))
    axes[0].imshow(rgb); axes[0].set_title("Tactical View")
    if len(all_points)>0:
        axes[0].scatter(all_points[:,0], all_points[:,1], c='lime', s=8, alpha=0.7)
    if len(risky_points)>0:
        axes[0].scatter(risky_points[:,0], risky_points[:,1], c='red', s=30, marker='x')
    masked = np.ma.masked_where(grid < 1, grid)
    axes[0].imshow(masked, cmap='jet', alpha=0.35, extent=(0,W,H,0))
    axes[0].axis('off')

    axes[1].imshow(heat_cluster, cmap='jet', vmin=0, vmax=vmax)
    axes[1].set_title("Cluster-based heat (10->40)")
    axes[1].axis('off')

    axes[2].imshow(heat_density, cmap='jet', vmin=0, vmax=vmax)
    axes[2].set_title("Density heat from ALL detections")
    axes[2].axis('off')

    plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()

def main():
    device = 'cpu' if FORCE_CPU else 'cpu'  # default CPU; change if you know GPU works: device='cuda'
    if FORCE_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model = load_model(YOLO_WEIGHTS, device=device)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("IMAGE NOT FOUND:", IMAGE_PATH); return
    H,W = img.shape[:2]

    all_pts = detect_centers(img, model, conf=CONF_THRESH, device=device)
    print("Detected points:", len(all_pts))

    labels, risky, eps = auto_dbscan(all_pts, base_min_samples=BASE_MIN_SAMPLES, eps_scale=EPS_SCALE)
    print("Auto chosen eps:", eps)
    print("Risky (cluster members):", len(risky))

    grid = points_to_grid(risky, H, W, rows=GRID_SIZE, cols=GRID_SIZE)
    print("Grid non-zero cells:", np.count_nonzero(grid), "Grid max:", grid.max())

    heat_d = density_heatmap(all_pts, H, W, rows=GRID_SIZE, cols=GRID_SIZE, scale=HEAT_SCALE)
    out_file = os.path.join(OUT_FOLDER, "result.png")
    visualize_and_save(img, all_pts, risky, grid, heat_d, out_file)
    print("Saved:", out_file)

if __name__ == "__main__":
    main()
