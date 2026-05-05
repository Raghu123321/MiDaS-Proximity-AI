import os
import cv2
import numpy as np
import torch
import requests
from flask import Flask, render_template, Response, send_from_directory, request, jsonify
import run 

app = Flask(__name__)

# Minimal Setup
UPLOAD_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
WEIGHTS_PATH = "weights/midas_v21_small_256.pt"
WEIGHTS_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("weights", exist_ok=True)

def download_weights():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Downloading weights from {WEIGHTS_URL}...")
        try:
            response = requests.get(WEIGHTS_URL, stream=True)
            response.raise_for_status()
            with open(WEIGHTS_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download weights: {e}")

# Cache Model
live_model = None
live_transform = None
latest_depth = 0.0 # Global to track proximity

def get_live_model():
    global live_model, live_transform
    if live_model is None:
        download_weights()
        device = torch.device("cpu")
        live_model, live_transform, _, _ = run.load_model(
            device, WEIGHTS_PATH, "midas_v21_small_256", optimize=False
        )
        live_model.eval()
    return live_model, live_transform

@app.route('/')
def index(): return render_template('index.html')

@app.route('/uploads/<path:f>')
def serve_u(f): return send_from_directory(UPLOAD_FOLDER, f)
@app.route('/outputs/<path:f>')
def serve_o(f): return send_from_directory(OUTPUT_FOLDER, f)

@app.route('/api/proximity')
def get_proximity():
    global latest_depth
    return {"depth": float(latest_depth)}

def process_single_frame(frame, model, transform):
    global latest_depth
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    img_input = transform({"image": img})["image"]

    with torch.no_grad():
        sample = torch.from_numpy(img_input).unsqueeze(0)
        prediction = model.forward(sample).squeeze().cpu().numpy()
        prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    h, w = prediction.shape
    center_val = np.mean(prediction[h//2-2:h//2+3, w//2-2:w//2+3])
    p_min, p_max = prediction.min(), prediction.max()
    current_u = (center_val - p_min) / (p_max - p_min + 1e-6)
    
    # Visual
    norm_vis = (prediction - p_min) / (p_max - p_min + 1e-6)
    depth_color = cv2.applyColorMap((norm_vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    
    return current_u, depth_color

@app.route('/api/process_frame', methods=['POST'])
def api_process_frame():
    global latest_depth
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    model, transform = get_live_model()
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    depth_score, depth_vis = process_single_frame(frame, model, transform)
    latest_depth = depth_score
    
    # Add UI indicators to the frame
    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (w//2, h//2), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 30, 2)
    u_text = f"Depth: {depth_score:.2f} U"
    cv2.putText(frame, u_text, (w//2 - 90, h//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(frame, u_text, (w//2 - 90, h//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    combined = np.concatenate((frame, depth_vis), axis=1)
    _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
    
    import base64
    img_str = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"image": img_str, "depth": float(depth_score)})

def generate_frames():
    global latest_depth
    model, transform = get_live_model()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Warning: Camera not found (this is normal on Render). Live stream via server will not work.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    last_depth_color = None
    smooth_u = 0.5
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        
        if frame_count % 3 == 0 or last_depth_color is None:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            img_input = transform({"image": img})["image"]

            with torch.no_grad():
                sample = torch.from_numpy(img_input).unsqueeze(0)
                prediction = model.forward(sample).squeeze().cpu().numpy()
                prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            # --- BEST MEASUREMENT (NORMALIZED U UNITS) ---
            # Max possible depth value for this model is roughly 1000
            # We normalize it so the score is always between 0 and 1
            h, w = prediction.shape
            center_val = np.mean(prediction[h//2-2:h//2+3, w//2-2:w//2+3])
            
            # Normalize based on dynamic range of current scene
            p_min, p_max = prediction.min(), prediction.max()
            current_u = (center_val - p_min) / (p_max - p_min + 1e-6)
            
            # Smoothing
            smooth_u = 0.8 * smooth_u + 0.2 * current_u
            latest_depth = smooth_u # Update global state

            # Depth Map Visual
            norm_vis = (prediction - p_min) / (p_max - p_min + 1e-6)
            last_depth_color = cv2.applyColorMap((norm_vis * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

        # UI Rendering
        h, w = frame.shape[:2]
        color = (0, 255, 255) # Bright Yellow
        cv2.drawMarker(frame, (w//2, h//2), color, cv2.MARKER_TILTED_CROSS, 30, 2)
        
        # Render Depth Score in U Units
        u_text = f"Depth: {smooth_u:.2f} U"
        cv2.putText(frame, u_text, (w//2 - 90, h//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, u_text, (w//2 - 90, h//2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        combined = np.concatenate((frame, last_depth_color), axis=1)
        _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
