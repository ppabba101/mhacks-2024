from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model only once during startup
model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

def scale_image(img, max_size=512):
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return img

def estimate_depth(img):
    height, width = img.shape[:2]
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()

def process_depth_map(depth_map):
    normalized_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    blurred_map = cv2.GaussianBlur(normalized_map, (9, 9), 0)
    
    height = blurred_map.shape[0]
    weights = np.linspace(1, 0.1, height)[:, np.newaxis]
    weighted_map = blurred_map * weights
    weighted_mean_per_column = np.sum(weighted_map, axis=0) / np.sum(weights)
    
    # Convert to a list of floats with 4 decimal places precision
    return [round(float(x), 4) for x in weighted_mean_per_column.tolist()]
    
def process_center_column(depth_map):
    normalized_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    blurred_map = cv2.GaussianBlur(normalized_map, (9, 9), 0)
    
    height, width = blurred_map.shape
    center_column = blurred_map[:, width // 2]  # Get the center column
    
    weights = np.linspace(1, 0.1, height)[:, np.newaxis]
    weighted_column = center_column * weights.flatten()  # Apply the weights to the center column
    
    weighted_mean_center_column = np.sum(weighted_column) / np.sum(weights)
    
    # Return the weighted mean for the center column rounded to 4 decimal places
    return round(float(weighted_mean_center_column), 4)


@app.route('/api/kmeans-depth', methods=['POST'])
def process_image_and_depth():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read image directly from file stream
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Scale down the image
    #img_scaled = scale_image(img)

    depth_map = estimate_depth(img)
    avg_pixels = process_depth_map(depth_map)

    response = jsonify({
        #"depth_map": depth_map.tolist(),
        "avg_pixels_per_column": avg_pixels#,
        #"original_size": img.shape[:2],
        #"scaled_size": img_scaled.shape[:2]
    })
    
    return response

def average_middle_third(avg_pixels):
    # Calculate the start and end indices for the middle third of the list
    third_len = len(avg_pixels) // 3
    start_index = third_len
    end_index = 2 * third_len

    # Get the middle third and calculate the average
    middle_third = avg_pixels[start_index:end_index]
    avg_middle_third = sum(middle_third) / len(middle_third)
    
    return avg_middle_third

@app.route('/api/center', methods=['POST'])
def process_image_and_depth1():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read image directly from file stream
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Scale down the image
    #img_scaled = scale_image(img)

    depth_map = estimate_depth(img)
    #avg_pixel_center = process_center_column(depth_map)
    avg_pixels = process_depth_map(depth_map)
    avg_pixel_center = average_middle_third(avg_pixels)
    
    response = jsonify({
        #"depth_map": depth_map.tolist(),
        "avg_pixels_per_column": avg_pixel_center#,
        #"original_size": img.shape[:2],
        #"scaled_size": img_scaled.shape[:2]
    })
    return response

if __name__ == "__main__":
    context = ('certificate.pem', 'privatekey.pem')  # Path to your certificate and key
    app.run(host="0.0.0.0", port=8775, ssl_context=context, debug=True)
    #app.run(host="0.0.0.0", port=8775)