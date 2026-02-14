# real_time_pipeline.py
import os, time
import cv2, numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage as ndi

# config
SOURCE = 0  # 0=webcam, or rtsp://user:pass@ip/stream
YOLO_WEIGHTS = "yolov8m.pt"
CONF = 0.30
FORCE_CPU = True
GRID_SIZE = 10
HEAT_SCALE = 4
DBSCAN_MIN = 4
EPS_SCALE = 1.2
OUT_FOLDER = "results"
os.makedirs(OUT_FOLDER, exist_ok=True)

if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = YOLO(YOLO_WEIGHTS)

def detect_centers_frame(frame, device='cpu'):
    res = model.predict(frame, conf=CONF, classes=0, device=device, verbose=False)
    pts = []
    for b in res[0].boxes:
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
        pts.append(((x1+x2)/2.0, (y1+y2)/2.0))
    return np.array(pts)

def auto_eps(points, k=4, scale=1.2):
    if len(points)==0 or len(points)<=1:
        return None
    k = min(k, len(points)-1)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances,_ = nbrs.kneighbors(points)
    eps = max(3.0, np.median(distances[:,1:k+1]) * scale)
    return eps

def run_rt(source=0):
    cap = cv2.VideoCapture(source)
    device = 'cpu'
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not read. Sleeping 1s and retrying.")
            time.sleep(1); continue
        H,W = frame.shape[:2]
        pts = detect_centers_frame(frame, device=device)
        eps = auto_eps(pts, scale=EPS_SCALE)
        labels = None; risky = np.array([])
        if eps is not None and len(pts) >= DBSCAN_MIN:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=DBSCAN_MIN).fit(pts)
            labels = clustering.labels_; risky = pts[labels!=-1]
        grid = np.zeros((GRID_SIZE, GRID_SIZE))
        cell_w = W/GRID_SIZE; cell_h = H/GRID_SIZE
        for p in risky:
            c = int(min(p[0] // cell_w, GRID_SIZE-1)); r=int(min(p[1]//cell_h, GRID_SIZE-1))
            grid[r,c]+=1
        heat = ndi.zoom(grid, HEAT_SCALE, order=3)
        # overlay tactical view on frame
        vis = frame.copy()
        for (x,y) in pts:
            cv2.circle(vis, (int(x),int(y)), 3, (0,255,0), -1)
        for (x,y) in risky:
            cv2.drawMarker(vis, (int(x),int(y)), (0,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
        # show
        cv2.imshow("Tactical", vis)
        # compress heat to an image and show side-by-side
        heat_norm = (heat / (heat.max()+1e-6) * 255).astype('uint8')
        heat_color = cv2.applyColorMap(cv2.resize(heat_norm, (W, H)), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(vis, 0.6, heat_color, 0.4, 0)
        cv2.imshow("Overlay", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_rt(SOURCE)
