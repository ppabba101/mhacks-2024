"""

Light Weight Implementation of MiDas Hybrid Model for Depth Estimation
Very little preprocessing and postprocessing for faster processing times
Meant for real-time applications where speed is more important than accuracy
And objects will normally be closer to the camera

"""
import cv2
import torch
import numpy as np
import cv2
import time

# Load MiDaS model and transformations
medium_model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
print(f"Loading MiDaS model: {medium_model_type}")
midas2 = torch.hub.load("intel-isl/MiDaS", medium_model_type)
device2 = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas2.to(device2)
midas2.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

# Function to estimate depth map using MiDaS

# Simple depth checks for faster processing times for more constant output
def estimate_depth1(img):
    height, width = img.shape[:2]
    if height > width: 
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device2)
    
    with torch.no_grad():
        prediction = midas2(input_batch)
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
    
    return [round(float(x), 4) for x in weighted_mean_per_column.tolist()]

def average_middle_third(avg_pixels):
    third_len = len(avg_pixels) // 3
    start_index = third_len
    end_index = 2 * third_len

    middle_third = avg_pixels[start_index:end_index]
    avg_middle_third = sum(middle_third) / len(middle_third)
    
    return avg_middle_third
