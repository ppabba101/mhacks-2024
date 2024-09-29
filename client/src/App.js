import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import { playRadar } from "./scripts/playRadar.js";
import { takePicture } from "./scripts/takePicture.js";
import VideoStream from "./scripts/VideoStream.js";

const PLAY_INTERVAL = 2;

function App() {
  const [radarPlaying, setRadarPlaying] = useState(false);
  const videoRef = useRef(null);

  const handleTakePicture = async () => {
    if (videoRef.current) {
      const data = await takePicture(videoRef.current);
      playRadar(data, PLAY_INTERVAL);
      return;
      const newWindow = window.open("", "_blank", "width=800,height=200");
      // Create a canvas element in the new window
      const canvas = newWindow.document.createElement("canvas");
      canvas.width = 800;
      canvas.height = 200;
      newWindow.document.body.appendChild(canvas);

      const ctx = canvas.getContext("2d");

      // Draw grayscale bar
      const barWidth = canvas.width / data.length;
      data.forEach((value, index) => {
        const grayValue = Math.floor(value * 255);
        ctx.fillStyle = `rgb(${grayValue},${grayValue},${grayValue})`;
        ctx.fillRect(index * barWidth, 0, barWidth, canvas.height);
      });
    } else {
      console.error("Video element not available");
    }
  };

  // useEffect(() => {
  //   setInterval(() => {
  //     handleTakePicture();
  //   }, PLAY_INTERVAL);
  // }, []);

useEffect(()=>{
document.onmousedown = handleTakePicture;
},[])

  return (
    <div>
      <VideoStream ref={videoRef} />
      
    </div>
  );
}

export default App;
