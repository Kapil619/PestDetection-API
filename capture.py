from flask import Flask, request, jsonify, send_from_directory
import cv2
import requests
import time
import os

app = Flask(__name__)

# Your Cloud Model API URL (Replace with actual URL)
MODEL_API_URL = "http://127.0.0.1:5000/predict"

STATIC_FOLDER = "static"
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)
# Function to capture an image using a webcam
def capture_and_send_image():
    camera = cv2.VideoCapture(1)  # 0 for the default webcam
    time.sleep(2)  # Give the camera time to warm up
    ret, frame = camera.read()
    if not ret:
        return {"error": "Failed to capture image"}
    
    timestamp = int(time.time()) 
    image_filename = f"image_{timestamp}.jpg"
    image_path = os.path.join(STATIC_FOLDER, image_filename)
    cv2.imwrite(image_path, frame)
    camera.release()

    with open(image_path, "rb") as img:
        files = {"image": img}
        response = requests.post(MODEL_API_URL, files=files)

    result =  response.json()  # Return the detection results
    result["image_url"] = f"http://{request.host}/static/{image_filename}"  # Add image URL

    return result

# API Endpoint to trigger the capture
@app.route("/capture", methods=["POST"])
def capture():
    result = capture_and_send_image()
    return jsonify(result)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)  # Runs on local network
