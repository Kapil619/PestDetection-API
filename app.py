from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Load your trained YOLOv8 model
model = YOLO("best.pt")  # adjust the path


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided to API"}), 400

    file = request.files["image"]
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Run inference
    results = model(image)

    # Process results (here, we format them as a list of detections)
    detections = []
    for result in results:
        for box in result.boxes.data.tolist():
            # box structure: [x_min, y_min, x_max, y_max, confidence, class]
            x_min, y_min, x_max, y_max, conf, cls = box
            detections.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": conf,
                "class": int(cls),
                "label": model.names[int(cls)]  # mapping class id to name
            })

    return jsonify({"detections": detections})


if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=5000, debug=True)
