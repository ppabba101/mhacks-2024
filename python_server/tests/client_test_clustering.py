import requests
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def resize_image(image_path, scale_factor=0.1):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image at {image_path}")
            return None
        
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dim = (width, height)
        
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpeg', resized_image)
        file_like_object = {'image': ('image.jpeg', buffer.tobytes(), 'image/jpeg')}
        return file_like_object

# Function to send image and get depth map, cluster information, and file paths
def send_image_and_get_kmeans_depth(image_path):
   url = 'https://150.136.71.52:8775/api/kmeans-depth'
   files = resize_image(image_path)
   response = requests.post(url, files=files, verify = False)


   if response.status_code == 200:
       data = response.json()
    #    objects_info = data['objects']
    #    annotated_image_path = data['annotated_image']
    #    depth_map_heatmap_path = data['depth_map_heatmap']
       avg_pixels_per_column = np.array(data['avg_pixels_per_column'])
       #return objects_info, annotated_image_path, depth_map_heatmap_path, avg_pixels_per_column
       return avg_pixels_per_column
   else:
       print(f"Error: {response.status_code}")
       print(response.text)
       return None, None, None, None


# Function to save the annotated image from the server
def fetch_annotated_image():
   url = 'https://improved-rotary-phone-pqv5jvwjg57h7xv9-8775.app.github.dev/api/get-image'
   response = requests.get(url, stream=True)


   if response.status_code == 200:
       with open('annotated_image.jpg', 'wb') as out_file:
           out_file.write(response.content)
       print("Annotated image saved as 'annotated_image.jpg'")
   else:
       print(f"Error fetching annotated image: {response.status_code}")


# Function to save the depth map heatmap from the server
def fetch_depth_map_heatmap():
   url = 'https://improved-rotary-phone-pqv5jvwjg57h7xv9-8775.app.github.dev/api/get-heatmap'
   response = requests.get(url, stream=True)


   if response.status_code == 200:
       with open('depth_map_heatmap.png', 'wb') as out_file:
           out_file.write(response.content)
       print("Depth map heatmap saved as 'depth_map_heatmap.png'")
   else:
       print(f"Error fetching depth map heatmap: {response.status_code}")


# Function to visualize average pixels per column
def visualize_avg_pixels(avg_pixels_per_column):
   plt.plot(avg_pixels_per_column, 'r-')
   plt.title('Average Pixel per Column')
   plt.xlabel('Column Index')
   plt.ylabel('Average Pixel Value')
   plt.tight_layout()
   plt.show()


if __name__ == "__main__":
    import time
    # Specify the path to your image
    image_path = os.path.expanduser("../IMAGES/test8.jpeg")

    # Check if the image exists
    time_start = time.time()
    if os.path.exists(image_path):
        # Send the image to the server and get response data
        #objects_info, annotated_image_path, depth_map_heatmap_path, avg_pixels_per_column = send_image_and_get_kmeans_depth(image_path)
        avg_pixels_per_column = send_image_and_get_kmeans_depth(image_path)
        
        # if objects_info is not None:
        #     # Display object info
        #     print("Objects detected with their depth values:")
        #     for obj in objects_info:
        #         print(f"Cluster ID: {obj['cluster_id']}, Center: ({obj['center_x']}, {obj['center_y']}), Depth Value: {obj['depth_value']}")
        
        # # Fetch and save the annotated image and depth map heatmap
        # fetch_annotated_image()
        # fetch_depth_map_heatmap()

        # Visualize average pixels per column
        print(f"Time taken: {time.time() - time_start:.2f} seconds")
        visualize_avg_pixels(avg_pixels_per_column)
    
    else:
        print(f"Error: Image file {image_path} not found.")
