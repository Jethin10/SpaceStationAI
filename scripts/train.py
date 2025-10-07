import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")   # e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)  # try 640 first; later try 832 for more accuracy
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=-1)   # -1 = auto
    ap.add_argument("--name", default="y8n_640_baseline")
    ap.add_argument("--device", default="cpu")         # "cpu" or GPU index like "0"
    ap.add_argument("--workers", type=int, default=0)  # 0 on Windows to avoid DataLoader hangs
    # Deprecated in Ultralytics >=8.x; accepted but ignored to avoid crashes
    ap.add_argument("--hyp", default=None, help="Deprecated in Ultralytics; ignored.")
    args = ap.parse_args()

    if args.hyp:
        print("[INFO] --hyp is deprecated in Ultralytics and will be ignored by this script.")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project="runs/train",
        name=args.name,
        workers=args.workers,
        patience=30,
        cos_lr=True,
        amp=True,
        close_mosaic=10,
        seed=42,
    )
    # Validate on the val split after training
    model.val(data=args.data, imgsz=args.imgsz, device=args.device, split="val")

if __name__ == "__main__":
    main()