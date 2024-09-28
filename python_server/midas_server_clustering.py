from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
from sklearn.cluster import KMeans
from flask_cors import CORS
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize MiDaS model for depth estimation
model_type = "DPT_Hybrid"
print(f"Loading MiDaS model: {model_type}")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load the necessary image transformations for MiDaS
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Function to estimate depth map using MiDaS
def estimate_depth(image_path):
    img = cv2.imread(image_path)
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

    depth_map = prediction.cpu().numpy()
    return depth_map

# Function to apply K-means clustering on the depth map
def apply_kmeans_clustering(depth_map, num_clusters=3):
    # Reshape the depth map into a 2D array where each pixel is a point
    pixels = depth_map.reshape(-1, 1)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    
    # Reshape the labels back to the depth map's shape
    clustered_map = kmeans.labels_.reshape(depth_map.shape)
    
    # Get the depth values corresponding to each cluster
    cluster_centers = kmeans.cluster_centers_
    
    return clustered_map, cluster_centers

# Function to calculate the center of mass (centroid) of each cluster
def calculate_cluster_centroids(clustered_map):
    centroids = []
    for cluster_id in np.unique(clustered_map):
        mask = clustered_map == cluster_id
        y_coords, x_coords = np.where(mask)
        
        if len(y_coords) > 0 and len(x_coords) > 0:
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            centroids.append((center_x, center_y, cluster_id))
    
    return centroids

@app.route('/api/kmeans-depth', methods=['POST'])
def process_image_and_kmeans_clusters():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = f"./image.jpg"
    file.save(file_path)

    # Generate depth map using MiDaS
    depth_map = estimate_depth(file_path)

    # Apply K-means clustering on the depth map
    num_clusters = 3  # Adjust this value for more or fewer clusters
    clustered_map, cluster_centers = apply_kmeans_clustering(depth_map, num_clusters)

    # Calculate the centroids (center of mass) of each cluster
    centroids = calculate_cluster_centroids(clustered_map)

    # Prepare the response data
    response_data = []
    for centroid in centroids:
        center_x, center_y, cluster_id = centroid
        depth_value = cluster_centers[cluster_id][0]  # Average depth value of the cluster
        response_data.append({
            "cluster_id": int(cluster_id),
            "center_x": center_x,
            "center_y": center_y,
            "depth_value": float(depth_value)
        })

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8775, debug=True)