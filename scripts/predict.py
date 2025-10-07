import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="runs/train/y8n_640_baseline/weights/best.pt")
ap.add_argument("--source", default="datasets/falcon_space/images/test")  # folder or a single image path
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--iou", type=float, default=0.6)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--device", default=0)
args = ap.parse_args()

model = YOLO(args.weights)
model.predict(
    source=args.source, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
    device=args.device, save=True, save_txt=True, save_conf=True,
    project="runs/predict", name="baseline"
)