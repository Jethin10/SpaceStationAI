from ultralytics import YOLO
m = YOLO("runs/train/y8s_832_falcon/weights/best.pt")  # change to your best run
m.export(format="onnx", opset=12, dynamic=True)