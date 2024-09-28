import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def send_image_and_get_depth(image_path):
    url = 'http://localhost:8775/api/depth'
    files = {'image': open(image_path, 'rb')}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        depth_map = np.array(data['depth_map'])
        closest_pixels_normalized_squared = np.array(data['closest_pixels_per_column_normalized_squared'])
        return depth_map, closest_pixels_normalized_squared
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None, None

def save_depth_map_as_image(depth_map, filename='depth_map.png'):
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    normalized_depth_map = 255 * (depth_map - min_val) / (max_val - min_val + 1e-6)
    depth_map_uint8 = normalized_depth_map.astype(np.uint8)
    cv2.imwrite(filename, depth_map_uint8)
    print(f"Depth map saved as {filename}")

def visualize_depth_and_closest(depth_map, closest_pixels_normalized_squared):
    plt.subplot(2, 1, 1)
    plt.imshow(depth_map, cmap='gray')
    plt.colorbar()
    plt.title('Depth Map')

    plt.subplot(2, 1, 2)
    plt.plot(closest_pixels_normalized_squared, 'r-')
    plt.title('Closest Pixel per Column (Normalized and Squared)')
    plt.xlabel('Column Index')
    plt.ylabel('Intensity (0-1)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = os.path.expanduser("~/mhacks-2024/python_server/room.jpeg")
    
    if os.path.exists(image_path):
        depth_map, closest_pixels_normalized_squared = send_image_and_get_depth(image_path)
        
        if depth_map is not None and closest_pixels_normalized_squared is not None:
            save_depth_map_as_image(depth_map, 'saved_depth_map.png')
            visualize_depth_and_closest(depth_map, closest_pixels_normalized_squared)
    else:
        print(f"Error: Image file {image_path} not found.")