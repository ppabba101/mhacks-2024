from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
import time

# Initialize Flask app
app = Flask(__name__)

# Load the MiDaS model for depth estimation
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
print(f"Loading model: {model_type}")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
print(f"Model loaded and moved to device: {device}")

# Load the necessary image transformations
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # Using small_transform for the MiDaS_small model

# Function to perform depth estimation
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

# Route for GET request - returns a simple message
@app.route('/api', methods=['GET'])
def api_home():
    return jsonify({"message": "Welcome to the Depth Estimation API"})

# Route for POST request - processes image file and returns depth map as a list
@app.route('/api/depth', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the image to a temporary path
    file_path = f"./{file.filename}"
    file.save(file_path)

    # Perform depth estimation
    depth_map = estimate_depth(file_path)

    # Convert the depth map to a serializable format (e.g., list)
    depth_map_serializable = depth_map.tolist()

    return jsonify({"depth_map": depth_map_serializable})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8775, debug=True)