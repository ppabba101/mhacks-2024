from flask import Flask, request, jsonify, send_file
import cv2
import torch
import numpy as np
from sklearn.cluster import KMeans
from flask_cors import CORS
import matplotlib.pyplot as plt
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize MiDaS model for depth estimation
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
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

# # Function to apply K-means clustering on the depth map
# def apply_kmeans_clustering(depth_map, num_clusters=8):
#     # Reshape the depth map into a 2D array where each pixel is a point
#     pixels = depth_map.reshape(-1, 1)
    
#     # Apply K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    
#     # Reshape the labels back to the depth map's shape
#     clustered_map = kmeans.labels_.reshape(depth_map.shape)
    
#     # Get the depth values corresponding to each cluster
#     cluster_centers = kmeans.cluster_centers_
    
#     return clustered_map, cluster_centers

# Function to apply K-means clustering on the depth map and spatial information
def apply_kmeans_with_spatial(depth_map, num_clusters=3):
    height, width = depth_map.shape

    # Generate coordinate grids (x and y) and flatten them
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Flatten the arrays to create feature vectors
    depth_flat = depth_map.flatten()
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    # Normalize depth and coordinates to the same range
    depth_normalized = (depth_flat - np.min(depth_flat)) / (np.max(depth_flat) - np.min(depth_flat) + 1e-6)
    x_normalized = x_flat / width
    y_normalized = y_flat / height

    # Stack depth and spatial information into a feature matrix
    features = np.stack([x_normalized, y_normalized, depth_normalized], axis=1)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)

    # Reshape the labels back to the depth map's shape
    clustered_map = kmeans.labels_.reshape(depth_map.shape)

    return clustered_map

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
# Function to colorize clusters for visualization
def colorize_clusters(clustered_map):
    height, width = clustered_map.shape
    colorized_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign random colors for each cluster
    num_clusters = len(np.unique(clustered_map))
    colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(num_clusters)]

    for cluster_id in range(num_clusters):
        mask = clustered_map == cluster_id
        colorized_image[mask] = colors[cluster_id]

    # Save the colorized cluster image
    colorized_image_path = './colorized_clusters.png'
    cv2.imwrite(colorized_image_path, colorized_image)
    
    return colorized_image_path

# Function to annotate and save the image with object locations and depths
def annotate_image_with_depth(image_path, clustered_map, centroids, cluster_centers):
    img = cv2.imread(image_path)
    
    # Loop through the centroids and annotate the image
    for centroid in centroids:
        center_x, center_y, cluster_id = centroid
        depth_value = cluster_centers[cluster_id][0]
        
        # Draw a circle at the centroid
        cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
        
        # Put the depth value next to the centroid
        cv2.putText(img, f"{depth_value:.2f}", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save the annotated image
    output_image_path = './annotated_image.jpg'
    cv2.imwrite(output_image_path, img)
    
    return output_image_path

# Function to visualize the depth map as a heatmap
def save_depth_map_heatmap(depth_map):
    plt.imshow(depth_map, cmap='plasma')  # Use 'plasma' colormap for heatmap
    plt.colorbar()  # Add a color bar for reference
    plt.title('Depth Map Heatmap')
    heatmap_path = './depth_map_heatmap.png'
    plt.savefig(heatmap_path)  # Save the heatmap as an image
    plt.close()
    return heatmap_path

# Function to plot average pixel values for each column
def avg_pixel_per_column(depth_map):
    exaggerated_map = cv2.GaussianBlur(depth_map, (9, 9), 0)  # Apply kernel
    
    mean_per_column = np.mean(exaggerated_map, axis=0)
    plt.plot(mean_per_column)  # Plot avg pixel values for each column
    plt.title("Average Pixel per Column")
    plt.xlabel("Column Index")
    plt.ylabel("Avg Pixel Value")
    avg_pixels_path = "./avg_pixels.png"
    plt.savefig(avg_pixels_path)  # Save as an image
    plt.close()
    
    return mean_per_column.tolist()

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
    num_clusters = 8  # Adjust this value for more or fewer clusters
    # clustered_map, cluster_centers = apply_kmeans_clustering(depth_map, num_clusters)
    clustered_map, cluster_centers = apply_kmeans_with_spatial(depth_map, num_clusters)

    # Calculate the centroids (center of mass) of each cluster
    centroids = calculate_cluster_centroids(clustered_map)

    # Create colorized image by groups
    colorized_image_path = colorize_clusters(clustered_map)

    # Annotate the image with object locations and depths
    annotated_image_path = annotate_image_with_depth(file_path, clustered_map, centroids, cluster_centers)

    # Save the depth map as a heatmap
    depth_map_heatmap_path = save_depth_map_heatmap(depth_map)

    # Get the average pixel per column
    avg_pixels = avg_pixel_per_column(depth_map)

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

    return jsonify({
        "objects": response_data,
        "annotated_image": annotated_image_path,
        "depth_map_heatmap": depth_map_heatmap_path,
        "avg_pixels_per_column": avg_pixels
    })

@app.route('/api/get-image', methods=['GET'])
def get_annotated_image():
    try:
        return send_file('./annotated_image.jpg', mimetype='image/jpeg')
    except FileNotFoundError:
        return jsonify({"error": "Annotated image not found"}), 404

@app.route('/api/get-heatmap', methods=['GET'])
def get_depth_map_heatmap():
    try:
        return send_file('./depth_map_heatmap.png', mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Depth map heatmap not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8775, debug=True)