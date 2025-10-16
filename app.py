from flask import Flask, request, jsonify
from flask_cors import CORS
from detect_yolo import detect_objects
from PIL import Image
import io, os, base64

app = Flask(__name__)
CORS(app)

os.makedirs("results", exist_ok=True)

@app.route('/')
def home():
    return "YOLOv8 Flask Object Detection API is running!"

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))

    # Save uploaded image temporarily
    temp_path = os.path.join("results", "temp.jpg")
    image.save(temp_path)

    # Run detection
    detections, results = detect_objects(temp_path)

    # Save output image with bounding boxes
    output_path = os.path.join("results", "output.jpg")
    results[0].save(filename=output_path)

    # Convert output image to base64 for quick preview (optional)
    with open(output_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    return jsonify({
        "detections": detections,
        "annotated_image": img_b64
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
