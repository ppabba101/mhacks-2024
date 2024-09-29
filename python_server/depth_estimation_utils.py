# depth_estimation_utils.py
import cv2
import torch
import numpy as np
from sklearn.cluster import KMeans
import time
import random
import matplotlib.pyplot as plt

# Load MiDaS model and transformations
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
print(f"Loading MiDaS model: {model_type}")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Function to estimate depth map using MiDaS
def estimate_depth(image_path):
    print("[INFO] Starting depth estimation process...")
    start_time = time.time()

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
    end_time = time.time()

    print(f"[INFO] Depth estimation completed in {end_time - start_time:.4f} seconds.")
    return depth_map

# Function to apply K-means clustering on the depth map with spatial information
def apply_kmeans_with_spatial(depth_map, num_clusters=3):
    print("[INFO] Starting K-means clustering process...")
    start_time = time.time()

    # Get original dimensions
    original_height, original_width = depth_map.shape

    # Calculate new width while maintaining aspect ratio
    fixed_height = 256
    aspect_ratio = original_width / original_height
    new_width = int(fixed_height * aspect_ratio)

    # Resize the depth map to fixed height (256) and new width
    resized_depth_map = cv2.resize(depth_map, (new_width, fixed_height), interpolation=cv2.INTER_LINEAR)

    # Create meshgrid for spatial information
    height, width = resized_depth_map.shape
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Flatten the arrays
    depth_flat = resized_depth_map.flatten()
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    # Normalize depth and coordinates
    depth_normalized = (depth_flat - np.min(depth_flat)) / (np.max(depth_flat) - np.min(depth_flat) + 1e-6)
    x_normalized = x_flat / width
    y_normalized = y_flat / height

    # Stack features for clustering
    features = np.stack([x_normalized, y_normalized, depth_normalized], axis=1)
    print(features.shape)
    print("[INFO] Running K-Means...")
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)

    # Reshape the clustered labels back to the resized depth map's shape
    clustered_map = kmeans.labels_.reshape(resized_depth_map.shape)
    cluster_centers = kmeans.cluster_centers_

    end_time = time.time()
    print(f"[INFO] K-means clustering completed in {end_time - start_time:.4f} seconds.")
    
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

# # Function to annotate and save the image with object locations and depths
# def annotate_image_with_depth(image_path, clustered_map, centroids, cluster_centers):
#     img = cv2.imread(image_path)
    
#     # Loop through the centroids and annotate the image
#     for centroid in centroids:
#         center_x, center_y, cluster_id = centroid
#         depth_value = cluster_centers[cluster_id][0]
        
#         # Draw a circle at the centroid
#         cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
        
#         # Put the depth value next to the centroid
#         cv2.putText(img, f"{depth_value:.2f}", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
#     # Save the annotated image
#     output_image_path = './annotated_image.jpg'
#     cv2.imwrite(output_image_path, img)
    
#     return output_image_path

# def annotate_image_with_depth(image_path, clustered_map, centroids, depth_map):
#     # Read the image
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"[ERROR] Image not found or unable to load: {image_path}")
#         return None
    
#     height, width = clustered_map.shape

#     # Ensure the image size matches the cluster map size (rescale if necessary)
#     img = cv2.resize(img, (width, height))

#     # Loop through the centroids and annotate the image
#     for centroid in centroids:
#         center_x, center_y, cluster_id = centroid
        
#         # Get the depth value from the depth map at the centroid location
#         depth_value = depth_map[center_y, center_x]
        
#         # Draw a circle at the centroid
#         cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)
        
#         # Put the depth value next to the centroid
#         cv2.putText(img, f"{depth_value:.2f}", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
#     # Save the annotated image
#     output_image_path = './annotated_image.jpg'
#     cv2.imwrite(output_image_path, img)
    
#     return output_image_path

def annotate_image_with_depth(image_path, clustered_map, centroids, depth_map):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Image not found or unable to load: {image_path}")
        return None

    # Get the sizes of the clustered map and the depth map
    clustered_height, clustered_width = clustered_map.shape
    depth_height, depth_width = depth_map.shape

    # Ensure the image size matches the cluster map size (rescale if necessary)
    img = cv2.resize(img, (clustered_width, clustered_height))

    # Adjust the depth map to match the clustered map size by resizing it
    resized_depth_map = cv2.resize(depth_map, (clustered_width, clustered_height), interpolation=cv2.INTER_LINEAR)

    # Loop through the centroids and annotate the image
    for centroid in centroids:
        center_x, center_y, cluster_id = centroid

        # Get the mask for the current cluster
        cluster_mask = (clustered_map == cluster_id)

        # Get the depth values for all the pixels within the cluster
        cluster_depth_values = resized_depth_map[cluster_mask]

        # Calculate the average depth of the cluster
        if len(cluster_depth_values) > 0:
            avg_depth_value = np.mean(cluster_depth_values)
        else:
            avg_depth_value = 0.0  # Handle edge case where there are no values

        # Draw a circle at the centroid
        cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)

        # Put the average depth value next to the centroid
        cv2.putText(img, f"{avg_depth_value:.2f}", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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