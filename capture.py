from flask import Flask, request, jsonify, send_from_directory, Response
import cv2
import requests
import time
import os

app = Flask(__name__)

# Your Cloud Model API URL (Replace with actual URL)
MODEL_API_URL = "http://65.1.91.195:5000/predict"

STATIC_FOLDER = "static"
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# Open the camera globally so we don't reopen it for each request
# (Make sure to adjust the index if needed)
camera = cv2.VideoCapture(0)  # Use 0 for default webcam if required

def generate_video_frames():
    """Generator function that continuously captures frames from the webcam."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            # Yield frame in byte format for MJPEG streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    """Endpoint for live video streaming."""
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def capture_and_send_image():
    """Captures a single frame from the camera, sends it to the model API, and returns the result."""
    # Capture a single frame (read once from the camera)
    ret, frame = camera.read()
    if not ret:
        return {"error": "Failed to capture image"}
    
    timestamp = int(time.time())
    image_filename = f"image_{timestamp}.jpg"
    image_path = os.path.join(STATIC_FOLDER, image_filename)
    cv2.imwrite(image_path, frame)

    # Send the captured image to the model API
    with open(image_path, "rb") as img:
        files = {"image": img}
        response = requests.post(MODEL_API_URL, files=files)

    result = response.json()  # Parse detection results
    # Add image URL to the result using the current host (e.g., Raspberry Pi's IP/domain)
    result["image_url"] = f"http://{request.host}/static/{image_filename}"
    return result

@app.route("/capture", methods=["POST"])
def capture():
    """Endpoint to capture a single image and return the detection results."""
    result = capture_and_send_image()
    return jsonify(result)

@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serves static files (captured images)."""
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    # Run the Flask server on all interfaces on port 5001
    app.run(host="0.0.0.0", port=5001)
