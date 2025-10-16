from ultralytics import YOLO
from PIL import Image

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

def detect_objects(image_path):
    """Run YOLOv8 inference on a single image and return detections."""
    results = model(image_path)
    detections = []

    for box in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        detections.append({
            "class_name": model.names[int(cls)],
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2]
        })
    return detections, results
