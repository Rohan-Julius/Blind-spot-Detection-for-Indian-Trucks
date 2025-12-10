import cv2
import numpy as np
import math
import argparse
from ultralytics import YOLO

# ---------- Box parameters (confirmed) ----------
TOP_HEIGHT_PX   = 80
RIGHT_TILT_DEG  = -1.50
KEEP_LEFT_VERTS = True
LINE_WIDTH      = 3
COLOR_BOX       = (0, 0, 255)  # red
COLOR_DET       = (0, 255, 0)  # green
FONT            = cv2.FONT_HERSHEY_SIMPLEX

# Bottom polygon (designed for 1920x1080)
BOTTOM_BASE = np.array([
    [79.01217054200174, 534.49691122714],   # P0
    [94.80535722822151, 882.0],             # P1 pivot
    [1476.7091922724514, 882.0],            # P2
    [426.46227763883667, 489.0914995042581] # P3
], dtype=np.float32)

FORCE_W, FORCE_H = 1920, 1080  # keep shape consistent

def rotate_about(pt, pivot, deg):
    th = math.radians(deg)
    R = np.array([[math.cos(th), -math.sin(th)],
                  [math.sin(th),  math.cos(th)]], dtype=np.float32)
    return (R @ (pt - pivot).astype(np.float32)) + pivot

def build_outline(bottom):
    top = bottom.copy()
    top[:,1] = bottom[:,1] - TOP_HEIGHT_PX
    pivot = bottom[1].copy()
    b = bottom.copy()
    b[2] = rotate_about(bottom[2], pivot, RIGHT_TILT_DEG)
    t = top.copy()
    t[1] = rotate_about(top[1], pivot, RIGHT_TILT_DEG)
    t[2] = rotate_about(top[2], pivot, RIGHT_TILT_DEG)
    if KEEP_LEFT_VERTS:
        t[0][0] = bottom[0][0]
        t[3][0] = bottom[3][0]
    return b, t

def draw_outline(frame, bottom, top, color=COLOR_BOX, lw=LINE_WIDTH):
    b = bottom.astype(int).reshape(-1,1,2)
    t = top.astype(int).reshape(-1,1,2)
    cv2.polylines(frame, [b], True, color, lw, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [t], True, color, lw, lineType=cv2.LINE_AA)
    for (bx,by),(tx,ty) in zip(bottom, top):
        cv2.line(frame, (int(bx),int(by)), (int(tx),int(ty)), color, lw, lineType=cv2.LINE_AA)

def center_in_poly(pt, poly):
    return cv2.pointPolygonTest(poly.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ultralytics YOLO model path, e.g., yolov8n.pt")
    ap.add_argument("--source", default="0", help="camera index (e.g., 0) or video/image path")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    ap.add_argument("--classes", default="", help="comma list to keep (e.g., car,truck,bus)")
    ap.add_argument("--save", default="", help="optional output video path")
    args = ap.parse_args()

    keep = set([s.strip().lower() for s in args.classes.split(",") if s.strip()]) \
            if args.classes else None

    # Build fixed overlay once
    bottom, top = build_outline(BOTTOM_BASE)
    gate_poly = bottom  # use bottom polygon for inside test

    # Open source
    src = args.source
    is_cam = src.isdigit()
    if is_cam:
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FORCE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FORCE_H)
    else:
        cap = cv2.VideoCapture(src)

    # Confirm size
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W == 0 or H == 0:
        print("Failed to open source."); return
    if (W,H) != (FORCE_W,FORCE_H) and is_cam:
        print(f"Warning: camera is {W}x{H}, expected 1920x1080 to exactly match box geometry.")

    # Model
    yolo = YOLO(args.model, task='detect')
    label_map = yolo.names

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(args.save, fourcc, fps, (W,H))

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Draw box overlay (no resizing)
        draw_outline(frame, bottom, top)

        # Run YOLO on the original frame (no masking so boxes draw outside too)
        res = yolo(frame, verbose=False, conf=args.conf)
        dets = []
        for b in res[0].boxes:
            cls_i = int(b.cls.item())
            name = label_map[cls_i].lower() if isinstance(label_map, dict) else str(label_map[cls_i]).lower()
            if keep and name not in keep: 
                continue
            x1,y1,x2,y2 = b.xyxy.cpu().numpy().squeeze().astype(int).tolist()
            cx, cy = (x1+x2)//2, (y1+y2)//2
            inside = center_in_poly((cx,cy), gate_poly)
            color = (0,0,255) if inside else COLOR_DET  # red if inside, green otherwise
            cv2.rectangle(frame,(x1,y1),(x2,y2), color, 2)
            cv2.circle(frame,(cx,cy), 3, color, -1)
            tag = f"{name} {float(b.conf.item()):.2f}"
            if inside:
                tag += " IN"
            cv2.putText(frame, tag, (x1, max(18,y1-6)), FONT, 0.55, color, 2, lineType=cv2.LINE_AA)

        cv2.imshow("YOLO inside box", frame)
        if writer is not None:
            writer.write(frame)
        k = cv2.waitKey(1) & 0xFF
        if k in [ord('q'), ord('Q')]: break

    cap.release()
    if writer is not None: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
