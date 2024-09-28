from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
import time
from flask_cors import CORS
import uuid

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model_type = "MiDaS_small"
print(f"Loading model: {model_type}")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
print(f"Model loaded and moved to device: {device}")

# Load the necessary image transformations
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

def estimate_depth(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        prediction = midas(input_batch)
        end_time = time.time()
        print(f"Prediction completed in {end_time - start_time:.4f} seconds")
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output

def apply_kernel_and_exaggerate(depth_map):
    blurred_depth_map = cv2.GaussianBlur(depth_map, (9, 9), 0)
    exaggerated_map = depth_map - blurred_depth_map
    min_val = np.min(exaggerated_map)
    max_val = np.max(exaggerated_map)
    normalized_exaggerated_map = (exaggerated_map - min_val) / (max_val - min_val + 1e-6)
    return normalized_exaggerated_map

def closest_pixel_per_column_normalized_squared(depth_map):
    exaggerated_map = apply_kernel_and_exaggerate(depth_map)
    closest_per_column = np.min(exaggerated_map, axis=0)
    squared_closest = np.square(closest_per_column)
    return squared_closest.tolist()

def save_depth_map(depth_map, file_path):
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    normalized_depth_map = 255 * (depth_map - min_val) / (max_val - min_val + 1e-6)
    depth_map_uint8 = normalized_depth_map.astype(np.uint8)
    cv2.imwrite(file_path, depth_map_uint8)

@app.route('/api/depth', methods=['POST'])
def process_image_and_closest_pixel():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = f"./{str(uuid.uuid4())}_image.jpg"
    file.save(file_path)

    try:
        depth_map = estimate_depth(file_path)
    except Exception as e:
        return jsonify({"error": "Depth estimation failed", "details": str(e)}), 500

    save_depth_map(depth_map, f'./{str(uuid.uuid4())}_depth_map.png')

    closest_pixels_normalized_squared = closest_pixel_per_column_normalized_squared(depth_map)

    return jsonify({
        "depth_map": depth_map.tolist(),  # Returning full depth map
        "depth_map_summary": {
            "min": float(np.min(depth_map)),
            "max": float(np.max(depth_map)),
            "mean": float(np.mean(depth_map))
        },
        "closest_pixels_per_column_normalized_squared": closest_pixels_normalized_squared
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8775, debug=True)