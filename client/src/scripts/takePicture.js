// Start of Selection
import { LOCAL_SERVER_URL } from "./constants.js";

export const takePicture = async (videoElement, ONE) => {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  // Set canvas size to 10% of the video size
  canvas.width = videoElement.videoWidth * (ONE ? 0.2 : 1);
  canvas.height = videoElement.videoHeight * (ONE ? 0.2 : 1);

  // Draw the video frame onto the canvas, scaling it down by 90%
  context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  // Convert canvas to blob
  return new Promise((resolve, reject) => {
    canvas.toBlob(async (blob) => {
      // Create FormData and append the blob
      const formData = new FormData();
      formData.append("image", blob, "image.jpg");

      try {
        // POST the image to the server
        const response = await fetch(
          `${LOCAL_SERVER_URL}/api/${ONE ? "depth" : "center"}`,
          {
            method: "POST",
            body: formData,
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        let data = await response.json();
        data = data.avg_pixels_per_column;
        console.log("Success:", data);

        // Function to create a less abrupt curve
        const P = 10;
        const smoothStep = (x) => {
          return Math.pow(x, P) / (Math.pow(x, P) + Math.pow(1 - x, P));
        };
        let smoothedData = data;
        if (!ONE) {
          smoothedData = data.map(Math.round);
        } else {
          // Isolate the center N columns
          const N = 5; // This can be changed to any desired value
          const centerIndex = Math.floor(data.length / 2);
          const start = Math.max(0, centerIndex - Math.floor(N / 2));
          const end = Math.min(data.length, start + N);
          smoothedData = data.slice(start, end);
          const avg =
            smoothedData.reduce((a, b) => a + b, 0) / smoothedData.length;
          smoothedData = [avg];
        }
        // The server returns a JSON array
        resolve([smoothedData, data]);
      } catch (error) {
        console.error("Error:", error);
        reject(error);
      }
    }, "image/jpeg");
  });
};
