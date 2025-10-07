import gradio as gr
from ultralytics import YOLO

model = YOLO("runs/train/y8s_832_falcon/weights/best.pt")  # change to your best weights

def infer(img):
    res = model.predict(img, conf=0.35, iou=0.6, imgsz=832)
    return res[0].plot()

gr.Interface(fn=infer, inputs=gr.Image(type="numpy"), outputs="image",
             title="Space Station Safety Detector").launch()