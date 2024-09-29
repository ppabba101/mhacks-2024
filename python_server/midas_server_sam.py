from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
import time
from flask_cors import CORS
import uuid
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
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
    
    # Check if height is greater than width, and if so, rotate the image
    height, width = img.shape[:2]
    if height > width:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print(f"Rotated the image 90 degrees to the left as height ({height}) > width ({width})")

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
    exaggerated_map = cv2.GaussianBlur(depth_map, (9, 9), 0)
    min_val = np.min(exaggerated_map)
    max_val = np.max(exaggerated_map)
    normalized_exaggerated_map = (exaggerated_map - min_val) / (max_val - min_val + 1e-6)
    cv2.imwrite(f'./normalized_exaggerated_map.png', normalized_exaggerated_map)
    return normalized_exaggerated_map

def closest_pixel_per_column(depth_map):
    exaggerated_map = apply_kernel_and_exaggerate(depth_map)
    
    # Get the maximum depth value for each column
    closest_per_column = np.max(exaggerated_map, axis=0)
    
    # Square the closest values
    squared_closest = np.square(closest_per_column)
    
    # Normalize the squared values between 0 and 1
    min_val = np.min(squared_closest)
    max_val = np.max(squared_closest)
    
    # To prevent division by zero in case all values are the same
    normalized_closest = (squared_closest - min_val) / (max_val - min_val + 1e-6)
    
    return normalized_closest.tolist()

def avg_pixel_per_column(depth_map):
    exaggerated_map = apply_kernel_and_exaggerate(depth_map)
    
    # Create a weight array that decreases linearly from top to bottom
    height = exaggerated_map.shape[0]
    weights = np.linspace(1, 0.1, height)[:, np.newaxis]
    
    # Apply weights to the exaggerated map
    weighted_map = exaggerated_map * weights
    
    # Calculate the weighted mean for each column
    weighted_mean_per_column = np.sum(weighted_map, axis=0) / np.sum(weights)
    
    plt.plot(weighted_mean_per_column)  # Plot weighted avg pixel values for each column
    plt.title("Weighted Average Pixel per Column")
    plt.xlabel("Column Index")
    plt.ylabel("Weighted Avg Pixel Value")
    plt.savefig(f"./weighted_avg_pixels.png")  # Save as an image
    plt.close()
    
    return weighted_mean_per_column.tolist()

def normalize_depth_map(depth_map):
    # cv2.imwrite(f'./depth_map.png', depth_map)
    plt.imshow(depth_map, cmap='plasma')  # Use 'plasma' colormap for heatmap
    plt.colorbar()  # Add a color bar for reference
    plt.title('Depth Map Heatmap')
    plt.savefig(f'./depth_map_heatmap.png')  # Save the heatmap as an image
    plt.close()
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    normalized_depth_map = 255 * (depth_map - min_val) / (max_val - min_val + 1e-6)
    cv2.imwrite(f'./normalized_depth_map.png', normalized_depth_map)
    return normalized_depth_map
def find_middle_pings(avg_pixels):
    print("Starting find_middle_pings function")
    middle_pings = np.zeros(len(avg_pixels))

    group_start = 0
    group = []

    percent_threshold = 100  # Adjust this value to determine what's "significantly different" (e.g., 10%)
    print(f"Percent threshold set to: {percent_threshold}%")
    
    for i, pixel in enumerate(avg_pixels):
        if not group:
            group.append(pixel)
            print(f"Starting new group with pixel {i}: {pixel}")
        else:
            group_mean = np.mean(group)
            percent_diff = abs(pixel - group_mean) / group_mean * 100
            print(f"Pixel {i}: {pixel}, Group mean: {group_mean:.2f}, Percent diff: {percent_diff:.2f}%")
            
            if percent_diff <= percent_threshold:
                group.append(pixel)
                print(f"Added pixel {i} to current group")
            else:
                if(pixel > group_mean):
                    # Set the middle ping
                    middle_index = group_start + len(group) // 2
                    middle_pings[middle_index] = group_mean
                    print(f"Set middle ping at index {middle_index} to {group_mean:.2f}")
                
                # Reset the group
                group_start = i
                group = [pixel]
                print(f"Started new group at index {i}")
    
    # Handle the last group
    if group:
        middle_index = group_start + len(group) // 2
        middle_pings[middle_index] = np.mean(group)
        print(f"Handled last group: Set middle ping at index {middle_index} to {np.mean(group):.2f}")
    
    print("Finished find_middle_pings function")
    return middle_pings.tolist()
@app.route('/api/depth', methods=['POST'])
def process_image_and_closest_pixel():
    # Takes in the image and saves the image to image.jpg
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = f"./image.jpg"
    file.save(file_path)

    # Sets depth map to the results of the MiDaS model
    try:
        depth_map = estimate_depth(file_path)
    except Exception as e:
        return jsonify({"error": "Depth estimation failed", "details": str(e)}), 500
    
    mean_intensity = np.mean(depth_map)
    std_intensity = np.std(depth_map)
    min_intensity = np.min(depth_map)
    max_intensity = np.max(depth_map)

    print(f"Mean intensity: {mean_intensity:.2f}")
    print(f"Standard deviation: {std_intensity:.2f}")
    print(f"Minimum intensity: {min_intensity:.2f}")
    print(f"Maximum intensity: {max_intensity:.2f}")


    normalized_depth_map = normalize_depth_map(depth_map)

    # closest_pixels = closest_pixel_per_column(normalized_depth_map)
    avg_pixels = avg_pixel_per_column(normalized_depth_map)

    return jsonify({
        "depth_map": normalized_depth_map.tolist(),  # Returning full depth map
        "depth_map_summary": {
            "min": float(np.min(depth_map)),
            "max": float(np.max(depth_map)),
            "mean": float(np.mean(depth_map))
        },
        "closest_pixels_per_column": avg_pixels
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8775, debug=True)