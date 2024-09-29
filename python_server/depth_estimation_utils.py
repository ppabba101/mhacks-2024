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

def apply_kmeans_with_spatial_color(depth_map, image, num_clusters=3, color_weight=0.1, blur_ksize=(15, 15)):
    print("[INFO] Starting K-means clustering process...")
    start_time = time.time()

    # Get original dimensions
    original_height, original_width = depth_map.shape

    # Calculate new width while maintaining aspect ratio
    fixed_height = 256
    aspect_ratio = original_width / original_height
    new_width = int(fixed_height * aspect_ratio)

    # Resize the depth map and image to fixed height (256) and new width
    resized_depth_map = cv2.resize(depth_map, (new_width, fixed_height), interpolation=cv2.INTER_LINEAR)
    resized_image = cv2.resize(image, (new_width, fixed_height), interpolation=cv2.INTER_LINEAR)

    # Step 1: Apply Gaussian blur to the RGB channels of the image to smooth color surfaces
    blurred_image = cv2.GaussianBlur(resized_image, blur_ksize, 0)
    blurred_image_path = './blurred_image.png'
    cv2.imwrite(blurred_image_path, blurred_image)


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

    # Extract RGB channels from the blurred image
    r_channel = blurred_image[:, :, 0].flatten()
    g_channel = blurred_image[:, :, 1].flatten()
    b_channel = blurred_image[:, :, 2].flatten()

    # Normalize RGB values to be between 0 and 1, and reduce their influence by scaling
    r_normalized = (r_channel / 255.0) * color_weight
    g_normalized = (g_channel / 255.0) * color_weight
    b_normalized = (b_channel / 255.0) * color_weight

    # Stack features for clustering (including RGB values with reduced importance)
    features = np.stack([x_normalized, y_normalized, depth_normalized, r_normalized, g_normalized, b_normalized], axis=1)
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

# Function to apply K-means clustering on the depth map with spatial information
# def apply_kmeans_with_spatial(depth_map, num_clusters=3):
#     print("[INFO] Starting K-means clustering process...")
#     start_time = time.time()

#     # Get original dimensions
#     original_height, original_width = depth_map.shape

#     # Calculate new width while maintaining aspect ratio
#     fixed_height = 256
#     aspect_ratio = original_width / original_height
#     new_width = int(fixed_height * aspect_ratio)

#     # Resize the depth map to fixed height (256) and new width
#     resized_depth_map = cv2.resize(depth_map, (new_width, fixed_height), interpolation=cv2.INTER_LINEAR)

#     # Create meshgrid for spatial information
#     height, width = resized_depth_map.shape
#     y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
#     # Flatten the arrays
#     depth_flat = resized_depth_map.flatten()
#     x_flat = x_coords.flatten()
#     y_flat = y_coords.flatten()

#     # Normalize depth and coordinates
#     depth_normalized = (depth_flat - np.min(depth_flat)) / (np.max(depth_flat) - np.min(depth_flat) + 1e-6)
#     x_normalized = x_flat / width
#     y_normalized = y_flat / height

#     # Stack features for clustering
#     features = np.stack([x_normalized, y_normalized, depth_normalized], axis=1)
#     print(features.shape)
#     print("[INFO] Running K-Means...")
    
#     # Apply K-means clustering
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)

#     # Reshape the clustered labels back to the resized depth map's shape
#     clustered_map = kmeans.labels_.reshape(resized_depth_map.shape)
#     cluster_centers = kmeans.cluster_centers_

#     end_time = time.time()
#     print(f"[INFO] K-means clustering completed in {end_time - start_time:.4f} seconds.")
    
#     return clustered_map, cluster_centers

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

# Calculate the score based on tuning parameters (fine-tune this)
def calculate_centroid_score(centroid, cluster_size, avg_depth, image_height, max_depth):
    """
    Calculates a score for the centroid based on several metrics:
    - Depth (closer objects are weighted more heavily)
    - Cluster size (larger clusters are given more weight)
    - Vertical position in the image (objects lower in the image are prioritized)
    
    Args:
        centroid: (x, y) coordinates of the centroid.
        cluster_size: The number of pixels in the cluster (size of the cluster).
        avg_depth: Average depth of the cluster.
        image_height: The height of the image (used to weight based on vertical position).
        max_depth: The maximum depth considered relevant (objects beyond this are ignored).
    
    Returns:
        score: A floating point score for the centroid.
    """
    x, y, cluster_id = centroid

    # Weight based on depth (closer = higher score)
    depth_weight = avg_depth / 1750  # Normalize depth score (closer to 1 is more important)
    
    # Weight based on size (larger clusters = higher score)
    size_weight = cluster_size / (image_height * image_height)  # Normalize size based on image size
    
    # Weight based on vertical position (lower on screen = higher score)
    position_weight = 1 - (y / image_height)  # Normalize vertical position (lower in the image is less important)
    
    # Combine the weights into a total score
    total_score = (depth_weight * 0.6) + (size_weight * 0.1) + (position_weight * 0.3)  # Adjust weight ratios as needed
    
    return total_score


def filter_relevant_centroids(centroids, clustered_map, depth_map, min_score_threshold, min_depth):
    """
    Filters centroids that pass a certain scoring threshold based on depth, size, and position.
    
    Args:
        centroids: List of centroids [(x, y, cluster_id)].
        clustered_map: The cluster map that groups pixels into clusters.
        depth_map: The depth map for average depth calculation.
        min_score_threshold: The minimum score threshold for centroids to be considered relevant.
        max_depth: The maximum depth to consider an object relevant.
    
    Returns:
        List of relevant centroids [(x, y, cluster_id, avg_depth, score)].
    """
    height, width = depth_map.shape
    clustered_height, clustered_width = clustered_map.shape
    resized_depth_map = cv2.resize(depth_map, (clustered_width, clustered_height), interpolation=cv2.INTER_LINEAR)
    relevant_centroids = []
    centroid_averages = []
    max_depth = np.max(depth_map)

    for centroid in centroids:
        x, y, cluster_id = centroid
        
        # Get the pixels corresponding to this cluster
        cluster_mask = clustered_map == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        # Calculate the average depth of the cluster
        avg_depth = np.mean(resized_depth_map[cluster_mask])
        
        # Add avg_depth to centroid_averages list for further calculations
        centroid_averages.append(avg_depth)

    # Step 1: Compute 3/4 of the average of all centroid averages
    if min_depth == 0:
        avg_of_centroid_averages = np.mean(centroid_averages)
        min_depth = avg_of_centroid_averages
        print(f"[INFO] min_depth (5/4 of average of centroid averages): {min_depth:.2f}")

    # Step 2: Filter centroids based on the min_depth
    for centroid in centroids:
        x, y, cluster_id = centroid
        
        # Get the pixels corresponding to this cluster
        cluster_mask = clustered_map == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        # Calculate the average depth of the cluster
        avg_depth = np.mean(resized_depth_map[cluster_mask])
        
        # Ignore clusters with avg_depth less than calculated min_depth
        if avg_depth < min_depth:
            continue
        
        # Calculate the score for this centroid
        score = calculate_centroid_score(centroid, cluster_size, avg_depth, height, max_depth)
        
        # Only pass centroids with a score above the threshold
        if score >= min_score_threshold:
            relevant_centroids.append((x, y, cluster_id))
    
    return relevant_centroids

# # Also needs fine-tuning for min_score_threshold and min_depth
# def filter_relevant_centroids(centroids, clustered_map, depth_map, min_score_threshold, min_depth):
#     """
#     Filters centroids that pass a certain scoring threshold based on depth, size, and position.
    
#     Args:
#         centroids: List of centroids [(x, y, cluster_id)].
#         clustered_map: The cluster map that groups pixels into clusters.
#         depth_map: The depth map for average depth calculation.
#         min_score_threshold: The minimum score threshold for centroids to be considered relevant.
#         max_depth: The maximum depth to consider an object relevant.
    
#     Returns:
#         List of relevant centroids [(x, y, cluster_id, avg_depth, score)].
#     """
#     height, width = depth_map.shape
#     clustered_height, clustered_width = clustered_map.shape
#     resized_depth_map = cv2.resize(depth_map, (clustered_width, clustered_height), interpolation=cv2.INTER_LINEAR)

#     relevant_centroids = []
#     max_depth = np.max(depth_map)
#     for centroid in centroids:
#         x, y, cluster_id = centroid
        
#         # Get the pixels corresponding to this cluster
#         cluster_mask = (clustered_map == cluster_id)
#         cluster_size = np.sum(cluster_mask)
        
#         # Calculate the average depth of the cluster
#         avg_depth = np.mean(resized_depth_map[cluster_mask])
        
#         # Ignore objects less than min_depth
#         if avg_depth < min_depth:
#             continue
        
        
#         # Calculate the score for this centroid
#         score = calculate_centroid_score(centroid, cluster_size, avg_depth, height, max_depth)
        
#         # Only pass centroids with a score above the threshold
#         if score >= min_score_threshold:
#             # Append the centroid in the format (x, y, cluster_id, avg_depth, score)
#             relevant_centroids.append((x, y, cluster_id))
    
#     return relevant_centroids


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

def annotate_image_with_depth(image_path, clustered_map, centroids, depth_map, relevant=False):
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
    if (relevant):
        output_image_path = './relevant_annotated_image.jpg'
    
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

def avg_pixel_per_column_with_centroid_weighting(depth_map, centroids, clustered_map, spread=60, edge_range=30):
    """
    Adjust the average pixel value for each column based on the centroid's position,
    resizing the exaggerated map to match the dimensions of the clustered map.
    Decrease the value of columns next to the edges of clusters unless there is another centroid 
    within a certain range of the edge.
    """
    # Step 1: Apply Gaussian blur to emphasize blobs
    exaggerated_map = cv2.GaussianBlur(depth_map, (9, 9), 0)
    
    # Step 2: Resize exaggerated map to match the dimensions of the clustered map
    clustered_height, clustered_width = clustered_map.shape
    exaggerated_map_resized = cv2.resize(exaggerated_map, (clustered_width, clustered_height), interpolation=cv2.INTER_LINEAR)
    
    # Step 3: Calculate the initial average pixel value per column on the resized map
    mean_per_column = np.mean(exaggerated_map_resized, axis=0)
    
    # Step 4: Modify the average values per column based on the relevant centroids
    height, width = depth_map.shape

    # Step 5: Create a mask to track columns with clusters and store edges of each cluster
    columns_with_clusters = np.zeros(clustered_width, dtype=bool)
    cluster_edges = []  # List to store the edges of clusters (first and last columns)

    # Iterate over each centroid
    for centroid in centroids:
        center_x, center_y, cluster_id = centroid
        
        # Get the mask for the current cluster
        cluster_mask = (clustered_map == cluster_id)
        
        # Get columns covered by this centroid's cluster
        x_coords = np.where(cluster_mask)[1]  # Column indices where the cluster has pixels
        
        if len(x_coords) == 0:
            continue
        
        # Mark the columns that contain clusters
        columns_with_clusters[x_coords] = True
        
        # Find the first and last columns covered by the cluster
        first_column = np.min(x_coords)
        last_column = np.max(x_coords)
        
        # Store the cluster edges
        cluster_edges.append((first_column, last_column))
        
        # Calculate the mean pixel value for the current centroid's cluster on the resized map
        centroid_mean = np.mean(exaggerated_map_resized[cluster_mask])
        
        # Calculate the weight based on the vertical position of the centroid (lower = higher weight)
        weight = center_y / height
        
        # Apply the weighted average to the relevant columns
        for col in range(first_column, last_column + 1):
            # Count how many pixels of the centroid are in this column
            column_pixels = np.sum(cluster_mask[:, col])
            
            if column_pixels > 0:
                # Adjust weight based on the number of pixels in this column
                column_weight = weight * (column_pixels * 50 / np.sum(cluster_mask))

                # Apply the weighted average formula
                mean_per_column[col] = (1 - column_weight) * mean_per_column[col] + column_weight * centroid_mean

    # Step 6: Reduce value of columns near the edges of clusters if no other centroid is nearby
    reduction_factor = 0.7  # You can tune this value

    # Function to check if there is a nearby centroid within the range
    def is_centroid_near(column, centroids, edge_range):
        for centroid in centroids:
            center_x, center_y, cluster_id = centroid
            if abs(center_x - column) <= edge_range:
                return True
        return False

    # Step 6: Spread the reduction beyond the cluster edges, with decay
    reduction_factor_base = 0.25  # Base reduction factor for edge pixels
    spread_distance = spread  # Distance over which the reduction spreads beyond the edge
    decay_factor = 0.05  # Reduction decay per pixel as we move away from the cluster

    # Function to calculate the reduction factor based on distance from the centroid
    def get_reduction_factor(distance, spread_distance):
        return max(reduction_factor_base * (1 - (decay_factor * distance / spread_distance)), 0)

    # Loop over the edges of the clusters
    for first_column, last_column in cluster_edges:
        # Spread reduction to the left of the first column
        for col in range(max(0, first_column - spread_distance), first_column):
            distance_from_edge = first_column - col
            reduction_factor = get_reduction_factor(distance_from_edge, spread_distance)
            
            # Apply reduction if no nearby centroid
            if not is_centroid_near(col, centroids, edge_range):
                mean_per_column[col] *= (1 - reduction_factor)
        
        # Spread reduction to the right of the last column
        for col in range(last_column + 1, min(clustered_width, last_column + spread_distance + 1)):
            distance_from_edge = col - last_column
            reduction_factor = get_reduction_factor(distance_from_edge, spread_distance)
            
            # Apply reduction if no nearby centroid
            if not is_centroid_near(col, centroids, edge_range):
                mean_per_column[col] *= (1 - reduction_factor)

    # # Loop over the edges of the clusters
    # for first_column, last_column in cluster_edges:
    #     # Reduce the value of the columns to the left of the first_column (if no centroid is nearby)
    #     for col in range(max(0, first_column - spread), first_column):
    #         if not is_centroid_near(col, centroids, edge_range):
    #             # Further the column from the first_column, less effect it will have
    #             col_weight = (spread - (first_column - col))/spread
    #             mean_per_column[col] *= reduction_factor
        
    #     # Reduce the value of the columns to the right of the last_column (if no centroid is nearby)
    #     for col in range(last_column + 1, min(clustered_width, last_column + spread + 1)):
    #         if not is_centroid_near(col, centroids, edge_range):
    #             mean_per_column[col] *= reduction_factor

    # Plot the modified average pixel values per column
    plt.plot(mean_per_column)
    plt.title("Modified Average Pixel per Column with Edge Reduction")
    plt.xlabel("Column Index")
    plt.ylabel("Avg Pixel Value")
    avg_pixels_path = "./avg_pixels_modified_with_edge_reduction.png"
    plt.savefig(avg_pixels_path)
    plt.close()

    # Define the smoothing kernel (e.g., a simple moving average)
    window_size = 10  # You can adjust this for more or less smoothing
    kernel = np.ones(window_size) / window_size

    # Pad the array at the start and end to prevent drops
    pad_size = window_size // 2
    padded_mean_per_column = np.pad(mean_per_column, (pad_size, pad_size), mode='edge')

    # Apply np.convolve for smoothing (using 'valid' to avoid the edge effect)
    smoothed_mean_per_column = np.convolve(padded_mean_per_column, kernel, mode='valid')

    # Plot the smoothed values
    plt.plot(smoothed_mean_per_column)
    plt.title("Smoothed Average Pixel per Column (No Edge Drop)")
    plt.xlabel("Column Index")
    plt.ylabel("Avg Pixel Value")

    # Save the smoothed plot
    smoothed_avg_pixels_path = "./smoothed_avg_pixels_no_edge_drop.png"
    plt.savefig(smoothed_avg_pixels_path)
    plt.close()
    
    return mean_per_column.tolist()

# def avg_pixel_per_column_with_centroid_weighting(depth_map, centroids, clustered_map):
#     """
#     Adjust the average pixel value for each column based on the centroid's position,
#     resizing the exaggerated map to match the dimensions of the clustered map.
#     """
#     # Step 1: Apply Gaussian blur to emphasize blobs
#     exaggerated_map = cv2.GaussianBlur(depth_map, (9, 9), 0)
    
#     # Step 2: Resize exaggerated map to match the dimensions of the clustered map
#     clustered_height, clustered_width = clustered_map.shape
#     exaggerated_map_resized = cv2.resize(exaggerated_map, (clustered_width, clustered_height), interpolation=cv2.INTER_LINEAR)
    
#     # Step 3: Calculate the initial average pixel value per column on the resized map
#     mean_per_column = np.mean(exaggerated_map_resized, axis=0)
    
#     # Step 4: Modify the average values per column based on the relevant centroids
#     height, width = depth_map.shape

#     # Iterate over each centroid
#     for centroid in centroids:
#         center_x, center_y, cluster_id = centroid
        
#         # Get the mask for the current cluster
#         cluster_mask = (clustered_map == cluster_id)
        
#         # Get columns covered by this centroid's cluster
#         x_coords = np.where(cluster_mask)[1]  # Column indices where the cluster has pixels
        
#         columns_with_clusters[x_coords] = True

#         if len(x_coords) == 0:
#             continue
        
#         # Find the first and last columns covered by the cluster
#         first_column = np.min(x_coords)
#         last_column = np.max(x_coords)
        
#         # Calculate the mean pixel value for the current centroid's cluster on the resized map
#         centroid_mean = np.mean(exaggerated_map_resized[cluster_mask])
        
#         # Calculate the weight based on the vertical position of the centroid (lower = higher weight)
#         weight = 1 - (center_y / height)  # Normalize the weight based on vertical position
#         # weight = 0.5
        
#         # Step 5: Apply the weighted average to the relevant columns
#         for col in range(first_column, last_column + 1):
#             # Count how many pixels of the centroid are in this column
#             column_pixels = np.sum(cluster_mask[:, col])
            
#             if column_pixels > 0:
#                 # Adjust weight based on the number of pixels in this column
#                 column_weight = weight * (column_pixels * 50 / np.sum(cluster_mask))

#                 # Apply the weighted average formula
#                 mean_per_column[col] = (1 - column_weight) * mean_per_column[col] + column_weight * centroid_mean
#                 # mean_per_column[col] = (1 - weight) * mean_per_column[col] + weight * centroid_mean

#     # Step 6: Decrease the value for columns without clusters
#     reduction_factor = 0.5  # You can tune this value
#     for col in range(clustered_width):
#         if not columns_with_clusters[col]:
#             mean_per_column[col] *= reduction_factor  # Decrease the value of the column

#     # Plot the modified average pixel values per column
#     plt.plot(mean_per_column)
#     plt.title("Modified Average Pixel per Column")
#     plt.xlabel("Column Index")
#     plt.ylabel("Avg Pixel Value")
#     avg_pixels_path = "./avg_pixels_modified.png"
#     plt.savefig(avg_pixels_path)
#     plt.close()
    
#     return mean_per_column.tolist()