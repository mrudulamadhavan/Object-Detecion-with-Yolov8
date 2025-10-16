from ultralytics import YOLO
from PIL import Image

# Load pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

def detect_objects(image_path):
    results = model(image_path)  # Run inference
    detections = []
    for result in results[0].boxes.data.tolist():  # xyxy, conf, class
        x1, y1, x2, y2, conf, cls = result
        detections.append({
            "class_name": model.names[int(cls)],
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2]
        })
    return detections
