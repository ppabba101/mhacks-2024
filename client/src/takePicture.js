export const takePicture = (videoElement) => {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  // Set canvas size to half of the video size
  canvas.width = videoElement.videoWidth / 2;
  canvas.height = videoElement.videoHeight / 2;

  // Draw the video frame onto the canvas, scaling it down by 50%
  context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

  // Get the image data as a base64-encoded string
  const imageDataUrl = canvas.toDataURL("image/jpeg");

  // Log the image data to the console
  console.log("Captured image:", imageDataUrl);

  const w = window.open();
  w.document.write(`<img src="${imageDataUrl}" />`);

  return imageDataUrl;
};
