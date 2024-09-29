from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import time
from depth_estimation_utils import *

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Main route to process image and perform K-means clustering
@app.route('/api/kmeans-depth', methods=['POST'])
def process_image_and_kmeans_clusters():
    print("[INFO] Received image, starting processing...")
    start_time = time.time()

    if 'image' not in request.files:
        print("[ERROR] No image file part found.")
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']
    if file.filename == '':
        print("[ERROR] No selected image file.")
        return jsonify({"error": "No selected file"}), 400

    file_path = f"./image.jpg"
    file.save(file_path)
    print(f"[INFO] Image saved at {file_path}.")

    try:
        depth_map = estimate_depth(file_path)
    except Exception as e:
        print(f"[ERROR] Depth estimation failed: {str(e)}")
        return jsonify({"error": f"Depth estimation failed: {str(e)}"}), 500

    num_clusters = 6
    clustered_map, cluster_centers = apply_kmeans_with_spatial(depth_map, num_clusters)

    # Calculate the centroids (center of mass) of each cluster
    centroids = calculate_cluster_centroids(clustered_map)

    # Create colorized image by groups
    colorized_image_path = colorize_clusters(clustered_map)

    # Annotate the image with object locations and depths
    annotated_image_path = annotate_image_with_depth(file_path, clustered_map, centroids, depth_map)

    # Save the depth map as a heatmap
    depth_map_heatmap_path = save_depth_map_heatmap(depth_map)

    # Get the average pixel per column
    avg_pixels = avg_pixel_per_column(depth_map)

    end_time = time.time()
    print(f"[INFO] Total image processing completed in {end_time - start_time:.4f} seconds.")

    response_data = []
    for centroid in centroids:
        center_x, center_y, cluster_id = centroid
        depth_value = cluster_centers[cluster_id][0]  # Average depth value of the cluster
        response_data.append({
            "cluster_id": int(cluster_id),
            "center_x": center_x,
            "center_y": center_y,
            "depth_value": float(depth_value)
    })

    return jsonify({
        "objects": response_data,
        "annotated_image": annotated_image_path,
        "depth_map_heatmap": depth_map_heatmap_path,
        "avg_pixels_per_column": avg_pixels
    })

@app.route('/api/get-image', methods=['GET'])
def get_annotated_image():
    try:
        return send_file('./annotated_image.jpg', mimetype='image/jpeg')
    except FileNotFoundError:
        return jsonify({"error": "Annotated image not found"}), 404

@app.route('/api/get-heatmap', methods=['GET'])
def get_depth_map_heatmap():
    try:
        return send_file('./depth_map_heatmap.png', mimetype='image/png')
    except FileNotFoundError:
        return jsonify({"error": "Depth map heatmap not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8775, debug=True)