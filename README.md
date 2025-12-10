# Blind-Spot Detection for Indian Trucks — Real-time ROI-gated YOLO

**Real-time single-camera blind-spot detection for trucks using ROI gating and optional SORT+Kalman tracking.**
YOLOv11 (edge/tiny) deployed on Raspberry Pi 5 — ROI overlays, color-coded alerts, and trajectory ghosting for anticipatory warnings.

---

## Project Overview

Trucks have large lateral blind spots that cause accidents during lane changes, turns, reversing, and depot maneuvers. This project implements a **low-latency, single-camera, on-edge** system to detect vehicles entering a truck’s blind spot and to provide stable, actionable alerts to drivers.

The pipeline uses:

* **YOLOv11** (tiny/nano variant for edge) for object detection
* **ROI-gating** (perspective-aware polygon) to accept only relevant detections inside the truck’s danger area
* Optional **SORT-style IoU tracker** + **per-track constant-velocity Kalman filter** for smoothing and 1–2s ghost projections
* CLI + interactive ROI tuning widgets for fast deployment and calibration

---

## Key Features

* Real-time YOLOv11 inference on Raspberry Pi 5 (target: **20–22 FPS**)
* Perspective-correct ROI polygon generator (four bottom vertices + perspective top)
* Color-coded visualization: **red** boxes for detections inside ROI, **green** for outside
* Optional tracking + Kalman trajectory projection (1–2 s ghost) and min-dwell confirmation to reduce flicker
* CLI for camera/video recording, model and confidence threshold overrides, ROI presets
* Lightweight, designed for edge deployment with quantization/optimization hooks

---

## System Architecture

```
Camera (single fisheye/regular camera)
        ↓ (frame)
   Preprocessing (resize, normalize 640×640)
        ↓
   YOLOv11 detector → bounding boxes + class + confidence
        ↓
   ROI Gate (point-in-polygon on detection centers)
        ↓
   Optional Tracker (SORT IoU + Kalman per-track)
        ↓
   Alert Logic (min-dwell, confirmation frames)
        ↓
   Renderer (overlay ROIs, boxes, ghost projections)
        ↓
   Output (live display / recorded video / alert JSON)
```

---

## Dataset

* **Source:** Roboflow workspace (RJProjects — Vehicle Detection).
* **Classes (example):** car, truck, auto, police_vehicle, scooter, motorcycle (your dataset contains more rotation-aware classes).
* **Images:** ~4,000 images (≈4,250 labeled objects)
* **Split:** 70% train / 20% val / 10% test
* **Annotation:** YOLO TXT bounding boxes
* **Preprocessing:** resized/normalized to **640×640** for training and inference
* **Avg objects / image:** ≈ **3.0**

> Note: include your `data.yaml` (train/val/test paths + names) in `/dataset`.

---

## Results & Metrics (summary)

* **Selected model:** YOLOv11 (nano/tiny)
* **mAP@0.5:** **0.511**
* **Best F1:** **0.49** (@ confidence ≈ 0.28)
* **Precision (max):** 1.00 (high confidence)
* **Recall (max):** 0.68
* **Edge performance:** **~20–22 FPS** on Raspberry Pi 5 (inference-only, single-thread, model-dependent)
* **MobileNetV2 (aux classification baseline):** val acc after fine-tune **94.72%**

Use the [Evaluation](#evaluation--reproducing-metrics) section to reproduce these.

___

## ROI Overlay Visualization
Shows how the blind-spot region is defined on the input frame using a perspective-aware ROI polygon.
<img width="1029" height="564" alt="Screenshot 2025-12-10 at 8 15 17 PM" src="https://github.com/user-attachments/assets/74c6a2bf-dfd1-4241-a1c1-ca3b245d2673" />

## Object Detected Inside ROI
Demonstrates a vehicle entering the blind-spot zone, triggering an ROI hit and alert.

<img width="406" height="710" alt="Screenshot 2025-12-10 at 8 18 21 PM" src="https://github.com/user-attachments/assets/e2f55e04-6aa9-4f17-9d36-7ad742c02353" />

## No Object Inside ROI
Displays a clear blind-spot region with no vehicles present inside the ROI.

<img width="406" height="710" alt="Screenshot 2025-12-10 at 8 16 51 PM" src="https://github.com/user-attachments/assets/ded6b558-58ce-4319-bf53-5e26389ecfd0" />

---

## Getting Started

### Prerequisites


* Python 3.9+
* Raspberry Pi 5 (for edge deployment) — or Linux x86-64 for dev/testing
* CUDA/GPU not required for Pi deployment (run CPU-optimized model); for faster training/use GPU machine
* `pip` and optionally `virtualenv`

### Install (dev / local)

```bash
# clone
git clone https://github.com/<yourusername>/blindspot-truck-yolov11.git
cd blindspot-truck-yolov11

# create venv
python3 -m venv .venv
source .venv/bin/activate

# install python deps
pip install -r requirements.txt
```

`requirements.txt` should include (example):

```
opencv-python
torch           # appropriate torch build for your platform
ultralytics     # or path to YOLOv11 repo
numpy
shapely         # for polygon/point-in-polygon
filterpy        # kalman filter (or custom)
scipy
tqdm
flask (optional dashboard)
```

---

## Quick Run (camera / video)

### Run on camera:

```bash
python detect_w_roi.py --model ./weights/my_model.pt --source 0 --roi-config configs/roi_default.json --conf 0.28 --track
```

### Run on file:

```bash
python detect_w_roi.py --model ./weights/my_model.pt --source /path/to/video.mp4 --out results/output.mp4 --conf 0.28
```

Common `run.py` arguments:

* `--model` path to model weights (`.pt`)
* `--source` camera index or video file
* `--conf` confidence threshold (suggested 0.25–0.35)
* `--track` enable SORT+Kalman tracking (optional)
* `--roi-config` ROI polygon JSON (4 bottom points + top shift + tilt)
* `--out` save annotated video
  
---

## ROI Design & Tuning

ROI is defined as:

```json
{
  "bottom_vertices": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "top_shift": 0.25,
  "right_tilt_deg": 6,
  "line_thickness": 4,
  "color": [255,0,0]
}
```

* **bottom_vertices:** 4 points along truck base (in image pixel coords)
* **top_shift:** fraction of vertical image height to shift up to make the top edge (perspective)
* **right_tilt_deg:** small rotation on right side to mimic curb perspective
  Use `roi_tuner.py` to visually tune and save presets for different truck models / camera mounts.

**Gating logic:** use detection box center point → `shapely` `Point.within(Polygon)` to accept/reject.

---

## Tracking & Prediction Module

* **Assignment:** IoU-based matching (SORT-style) between detections and existing tracks
* **Per-track predictor:** 2D constant-velocity Kalman Filter (x, y, vx, vy)
* **Stabilization:** smooth box jitter and output ghost projection for 1–2 seconds ahead
* **Alert logic:** require `N_confirm` frames before alert; require `min_dwell` frames before clearing to prevent flicker

Config options:

```json
{
  "iou_thresh": 0.3,
  "max_age": 30,
  "min_hits": 3,
  "prediction_seconds": 1.5
}
```

---

## Optimizations for Raspberry Pi 5 (edge)

To get stable 20–22 FPS on Pi 5:

* Use the **tiny/nano** variant of YOLOv11 (`yolov11n.pt`) or export a pruned/quantized model.
* Run with `batch_size=1`, single inference thread.
* Convert weights to **ONNX** then to a runtime optimized for Pi (e.g., ONNX Runtime with NNAPI if available, or TFLite if you convert further).
* Use **8-bit quantization** (post-training static quantization) to reduce model size and improve throughput.
* Preprocess frames to 640×640, and avoid expensive visualizations when in production.
* Use `multiprocessing` or a producer-consumer pattern: capture → inference → render to keep camera capture steady.

---

## Limitations & Future Work

* **Class imbalance** (eg. police_vehicle dominating) reduces recall for rare classes (person, ambulance). Add more labeled samples for rare classes.
* **Small objects** (bikes, pedestrians) remain challenging — consider higher-res crops or multi-scale inference.
* **Single camera** cannot resolve occluded objects behind large trucks; multi-camera fusion would improve coverage.
* Next steps: quantization-aware training, model pruning, dataset balancing, Lidar/Radar fusion, and lane/ego-vehicle context for better alerts.

---

## Troubleshooting

* **Low FPS:** try the tiny model, disable tracking, turn off live GUI drawing.
* **Many false positives outside ROI:** confirm ROI coordinates and point-in-polygon logic use the detection *center*, not corner.
* **Alerts flicker:** increase `min_dwell` and `min_hits` in tracker; verify detection confidence threshold.
* **Model not loading on Pi:** ensure correct Torch / runtime build compatible with Pi (or use converted ONNX/TFLite).

