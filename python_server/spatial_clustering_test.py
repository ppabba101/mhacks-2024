# test_spatial_clustering.py
import os
import matplotlib.pyplot as plt
import numpy as np
from depth_estimation_utils import estimate_depth, apply_kmeans_with_spatial

# Function to visualize clusters on depth map
def visualize_clusters(clustered_map, depth_map):
    plt.figure(figsize=(10, 5))

    # Original depth map heatmap
    plt.subplot(1, 2, 1)
    plt.imshow(depth_map, cmap='plasma')
    plt.title("Depth Map Heatmap")
    plt.colorbar()

    # Clustered map visualization
    plt.subplot(1, 2, 2)
    plt.imshow(clustered_map, cmap='tab20')
    plt.title("K-means Clusters on Depth Map")
    plt.colorbar()

    plt.savefig('clustered_image_output.png')
    plt.show()

if __name__ == "__main__":
    # Path to the image
    image_path = os.path.expanduser("~/mhacks-2024/python_server/test2.jpeg")
    
    # Make sure the image exists
    if os.path.exists(image_path):
        # Estimate depth
        depth_map = estimate_depth(image_path)

        # Apply K-means clustering
        num_clusters = 3
        clustered_map = apply_kmeans_with_spatial(depth_map, num_clusters)

        # Visualize clusters and depth map
        visualize_clusters(clustered_map, depth_map)
    else:
        print(f"Error: Image file {image_path} not found.")