import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Function to send a GET request to check if the server is running
def check_server():
    url = 'http://localhost:8775/api'
    response = requests.get(url)

    if response.status_code == 200:
        print(f"Server is running: {response.json()['message']}")
    else:
        print("Failed to connect to server.")
    return response.status_code == 200

# Function to send image via POST request and receive depth map
def send_image(image_path):
    url = 'http://localhost:8775/api/depth'

    # Open the image file and send it via POST request
    with open(image_path, 'rb') as img_file:
        files = {'file': img_file}
        response = requests.post(url, files=files)

    # Print the raw response content for debugging
    print("Response status code:", response.status_code)
    print("Response content:", response.text)  # This will print the raw content of the response

    # Check if response contains valid JSON before trying to parse it
    if response.headers.get('Content-Type') == 'application/json':
        try:
            depth_map = np.array(response.json().get('depth_map', []), dtype=np.float32)
            print(f"Depth map received successfully with shape {depth_map.shape}")

            # Load the original image to get the dimensions
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape

            # Reshape the depth map to the image size (assuming server returns correct depth map)
            depth_map = depth_map.reshape((height, width))

            # Visualize the depth map
            plt.imshow(depth_map, cmap='gray')
            plt.colorbar()
            plt.title("Depth Map Visualization")
            plt.show()
        except Exception as e:
            print("Error parsing JSON response:", e)
    else:
        print(f"Error: Expected JSON response but got {response.headers.get('Content-Type')}")

if __name__ == "__main__":
    # Check if the server is running by making a GET request
    if check_server():
        # Path to the image you want to send
        image_path = os.path.expanduser("~/mhacks-2024/python_server/room.jpeg")  # Replace with the actual image path
        send_image(image_path)