from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import uuid
import time
from depth_estimation_utils import estimate_depth, apply_kmeans_with_spatial

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

    file_path = f"./{uuid.uuid4()}_image.jpg"
    file.save(file_path)
    print(f"[INFO] Image saved at {file_path}.")

    try:
        depth_map = estimate_depth(file_path)
    except Exception as e:
        print(f"[ERROR] Depth estimation failed: {str(e)}")
        return jsonify({"error": f"Depth estimation failed: {str(e)}"}), 500

    num_clusters = 3
    clustered_map = apply_kmeans_with_spatial(depth_map, num_clusters)

    end_time = time.time()
    print(f"[INFO] Total image processing completed in {end_time - start_time:.4f} seconds.")

    return jsonify({
        "message": "Processing complete",
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8775, debug=True)