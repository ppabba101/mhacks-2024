import { SERVER_URL } from "./constants.js";

export const takePicture = async (videoElement) => {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  // Set canvas size to half of the video size
  canvas.width = videoElement.videoWidth / 2;
  canvas.height = videoElement.videoHeight / 2;

  // Draw the video frame onto the canvas, scaling it down by 50%
  context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  // Convert canvas to blob
  return new Promise((resolve, reject) => {
    canvas.toBlob(async (blob) => {
      // Create FormData and append the blob
      const formData = new FormData();
      formData.append("image", blob, "image.jpg");

      try {
        // POST the image to the server
        const response = await fetch(`${SERVER_URL}/api/depth`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        let data = await response.json();
        data=data.closest_pixels_per_column;
        console.log("Success:", data);

        
  // Function to create a less abrupt curve
  const P = 8;
  const smoothStep = (x) => {
    return Math.pow(x,P)/(Math.pow(x,P)+Math.pow(1-x,P))
  };

  
  data=data.map(smoothStep);


        // The server returns a JSON array
        resolve(data);
      } catch (error) {
        console.error("Error:", error);
        reject(error);
      }
    }, "image/jpeg");
  });
};
