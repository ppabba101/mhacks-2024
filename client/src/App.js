import React, { useState, useRef } from "react";
import "./App.css";
import { playRadar } from "./scripts/playRadar.js";
import { takePicture } from "./scripts/takePicture.js";
import VideoStream from "./scripts/VideoStream.js";

function App() {
  const [radarPlaying, setRadarPlaying] = useState(false);
  const videoRef = useRef(null);

  const handlePlayRadar = () => {
    if (!radarPlaying) {
      setRadarPlaying(true);
      const volumes = [
        57, 57, 57, 58, 58, 58, 57, 57, 57, 57, 57, 57, 57, 56, 56, 55, 55, 56,
        56, 55, 55, 55, 55, 54, 54, 54, 53, 54, 53, 53, 53, 53, 53, 52, 52, 52,
        51, 51, 51, 51, 51, 51, 51, 51, 50, 50, 50, 50, 50, 50, 49, 49, 49, 48,
        48, 47, 47, 47, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 45, 45, 45,
        43, 43, 43, 43, 43, 43, 42, 42, 42, 41, 41, 40, 40, 40, 40, 40, 40, 39,
        38, 37, 36, 34, 34, 34, 35, 35, 37, 39, 41, 44, 44, 47, 63, 74, 77, 77,
        78, 79, 79, 80, 79, 78, 77, 77, 77, 77, 76, 76, 76, 77, 77, 77, 77, 77,
        77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 77, 77, 76, 76, 76, 75,
        75, 75, 74, 74, 74, 74, 74, 74, 74, 74, 74, 73, 67, 58, 55, 54, 54, 53,
        53, 53, 53, 53, 52, 52, 52, 52, 51, 51, 51, 51, 51, 50, 50, 50, 50, 50,
        49, 49, 49, 49, 48, 48, 48, 48, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47,
        47, 47, 47, 47, 47, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
        46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 45,
        45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 44, 44, 44, 44, 44, 44,
        44, 45, 45, 44, 44, 44, 44, 44, 44, 43, 43, 43, 44, 43, 43, 42, 42, 41,
        40, 41, 40, 41, 41, 42, 42, 42, 43, 42, 42, 42, 42, 42, 42, 42, 42, 41,
        41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 41, 41, 41, 41, 41, 40, 40,
        40, 39, 39, 39, 40, 39, 40, 39, 39, 40, 40, 40, 40, 39, 39, 39, 38, 36,
        34, 33, 31, 30, 30, 29, 29, 28, 28, 28, 28, 28, 27, 27, 27, 27, 27, 27,
        27, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 26, 26, 26, 26,
        26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 26, 26, 26, 26, 26,
        26, 26, 25, 25, 25, 25, 25, 25, 24, 24, 24, 24, 23, 23, 23, 23, 23, 23,
        22, 22, 22, 22, 21, 21, 21, 21, 21, 20, 20, 20, 20, 19, 19, 19, 19, 18,
        18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 16, 16, 16, 16, 16, 16, 15, 15,
        15, 15, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 12, 13, 12,
        12, 11, 11, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 10, 9, 8, 8, 8, 8, 8, 7,
        7, 7, 8, 8, 6, 5, 7, 7, 6, 7, 7, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8,
        9, 9, 9, 9, 9, 9, 8, 9, 8, 8, 7, 6, 6, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 4, 4, 4, 3, 3, 2,
        1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 2, 2, 2,
        3, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 4, 4, 3, 4, 6, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 8, 9, 9, 8, 9, 9, 9, 9, 10, 10, 10, 9,
      ];
      const duration = 1;
      playRadar(volumes, duration);
      setTimeout(() => setRadarPlaying(false), duration * 1000);
    }
  };
  const handleTakePicture = async () => {
    if (videoRef.current) {
      const data = await takePicture(videoRef.current);
      playRadar(data, 1);

      // // Open a new window
      // const newWindow = window.open("", "_blank", "width=800,height=200");

      // // Create a canvas element in the new window
      // const canvas = newWindow.document.createElement("canvas");
      // canvas.width = 800;
      // canvas.height = 200;
      // newWindow.document.body.appendChild(canvas);

      // const ctx = canvas.getContext("2d");

      // // Draw grayscale bar
      // const barWidth = canvas.width / data.length;
      // data.forEach((value, index) => {
      //   const grayValue = Math.floor(value * 255);
      //   ctx.fillStyle = `rgb(${grayValue},${grayValue},${grayValue})`;
      //   ctx.fillRect(index * barWidth, 0, barWidth, canvas.height);
      // });
    } else {
      console.error("Video element not available");
    }
  };

  return (
    <div>
      <VideoStream ref={videoRef} />
      <button id="radar" onClick={handlePlayRadar} disabled={radarPlaying}>
        {radarPlaying ? "Playing..." : "Play Radar"}
      </button>
      <button id="take-picture" onClick={handleTakePicture}>
        Take Picture
      </button>
    </div>
  );
}

export default App;
