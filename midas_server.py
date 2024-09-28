import cv2
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import websockets
import asyncio

# Load the MiDaS model for depth estimation
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
print(f"Loading model: {model_type}")
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
print(f"Model loaded and moved to device: {device}")

# Load the necessary image transformations
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform  # Using small_transform for the MiDaS_small model

# Function to perform depth estimation
def estimate_depth(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    
    with torch.no_grad():
        start_time = time.time()
        prediction = midas(input_batch)
        end_time = time.time()
        print(f"Prediction completed in {end_time - start_time:.4f} seconds")
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output

# Function to handle the web socket communication
async def handle_websocket(websocket, path):
    try:
        async for message in websocket:
            # Assuming 'message' contains the image path sent by the client
            print(f"Received image path: {message}")
            
            # Perform depth estimation
            depth_map = estimate_depth(message)
            
            # Convert the depth map to bytes to send back
            depth_map_bytes = depth_map.tobytes()
            await websocket.send(depth_map_bytes)
            print("Depth map sent back to client")
    
    except Exception as e:
        print(f"Error handling the websocket: {e}")

# Start the web socket server
async def start_server():
    async with websockets.serve(handle_websocket, "localhost", 8765):
        print("Web socket server started on ws://localhost:8765")
        await asyncio.Future()  # Run the server forever

# Main event loop to run the server
if __name__ == "__main__":
    asyncio.run(start_server())