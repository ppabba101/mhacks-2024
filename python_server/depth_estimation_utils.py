# depth_estimation_utils.py
import cv2
import torch
import numpy as np
from sklearn.cluster import KMeans
import time

# Load MiDaS model and transformations
model_type = "MiDaS_small"
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

    height, width = depth_map.shape
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    depth_flat = depth_map.flatten()
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()

    depth_normalized = (depth_flat - np.min(depth_flat)) / (np.max(depth_flat) - np.min(depth_flat) + 1e-6)
    x_normalized = x_flat / width
    y_normalized = y_flat / height

    features = np.stack([x_normalized, y_normalized, depth_normalized], axis=1)
    print("[INFO] Running KMeans...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)

    clustered_map = kmeans.labels_.reshape(depth_map.shape)

    end_time = time.time()
    print(f"[INFO] K-means clustering completed in {end_time - start_time:.4f} seconds.")
    return clustered_map