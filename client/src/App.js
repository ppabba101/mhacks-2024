import React, { useState, useRef } from "react";
import "./App.css";
import { playRadar } from "./playRadar.js";
import { takePicture } from "./takePicture.js";
import VideoStream from "./VideoStream.js";

function App() {
  const [radarPlaying, setRadarPlaying] = useState(false);
  const videoRef = useRef(null);

  const handlePlayRadar = () => {
    if (!radarPlaying) {
      setRadarPlaying(true);
      const volumes = [0, 0.5, 1, 0.5, 0];
      const duration = 2;
      playRadar(volumes, duration);
      setTimeout(() => setRadarPlaying(false), duration * 1000);
    }
  };

  const handleTakePicture = () => {
    if (videoRef.current) {
      takePicture(videoRef.current);
    } else {
      console.error("Video element not available");
    }
  };

  return (
    <div>
      <h1>Radar</h1>
      <VideoStream ref={videoRef} />
      <button onClick={handlePlayRadar} disabled={radarPlaying}>
        {radarPlaying ? "Playing..." : "Play Radar"}
      </button>
      <button onClick={handleTakePicture}>Take Picture</button>
    </div>
  );
}

export default App;
