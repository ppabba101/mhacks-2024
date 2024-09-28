import asyncio
import websockets
import numpy as np
import cv2
import matplotlib.pyplot as plt

async def send_image(image_path):
    uri = "ws://localhost:8765"
    
    async with websockets.connect(uri) as websocket:
        # Send image path to the server
        await websocket.send(image_path)
        print(f"Image path sent: {image_path}")
        
        # Receive the depth map bytes
        depth_map_bytes = await websocket.recv()
        print(f"Depth map received (length: {len(depth_map_bytes)} bytes)")
        
        # Get the image width and height from the
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        # Convert the bytes back to a numpy array
        depth_map = np.frombuffer(depth_map_bytes, dtype=np.float32).reshape((height, width))
        
        # Visualize the depth map
        plt.imshow(depth_map, cmap='gray')
        plt.colorbar()
        plt.title("Depth Map Visualization")
        plt.show()

if __name__ == "__main__":
    # Path to the image you want to send
    image_path = "path_to_your_image.jpg"  # Replace with the actual image path
    asyncio.run(send_image(image_path))