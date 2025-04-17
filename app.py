from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import os
import uuid  # for generating unique filenames

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static"  # or any folder of your choice
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

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

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    detections = []
    for result in results:
        for box in result.boxes.data.tolist():
            # box structure: [x_min, y_min, x_max, y_max, confidence, class]
            x_min, y_min, x_max, y_max, conf, cls = box
            detections.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "confidence": conf,
                "class": int(cls),
                "label": model.names[int(cls)]
            })
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, y_min), f"{model.names[int(cls)]} {conf:.2f}", fill="red")

    # Generate a unique filename to avoid overwriting existing files
    unique_filename = f"{uuid.uuid4()}.jpg"
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    image.save(output_path)  # save bounding-box image to disk

    # Build a URL for the saved image
    image_url = f"/static/{unique_filename}"

    # Return the detections and the image URL in JSON
    return jsonify({
        "detections": detections,
        "image_url": image_url
    })

# Optional: Serve files from "/static" (Flask automatically does this if app.static_folder is set)
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    print("Starting server...")
    app.run(host="0.0.0.0", port=5000, debug=False)
