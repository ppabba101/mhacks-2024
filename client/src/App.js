import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import { playRadar } from "./scripts/playRadar.js";
import { takePicture } from "./scripts/takePicture.js";
import VideoStream from "./scripts/VideoStream.js";

const PLAY_INTERVAL = 1.5;
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function App() {
  const [ONE, setONE] = useState(false);
  const [started, setStarted] = useState(false);
  const videoRef = useRef(null);
  const oscillatorRef = useRef(null);
  const gainNodeRef = useRef(null);

  useEffect(() => {
    if (ONE) {
      oscillatorRef.current = audioCtx.createOscillator();
      gainNodeRef.current = audioCtx.createGain();

      oscillatorRef.current.type = "sine";
      oscillatorRef.current.connect(gainNodeRef.current);
      gainNodeRef.current.connect(audioCtx.destination);

      oscillatorRef.current.start();
      gainNodeRef.current.gain.setValueAtTime(0.5, audioCtx.currentTime);
    }

    return () => {
      if (ONE && oscillatorRef.current) {
        oscillatorRef.current.stop();
        oscillatorRef.current.disconnect();
        gainNodeRef.current.disconnect();
      }
    };
  }, [ONE]);

  useEffect(() => {
    console.log("THIS EFFECT IS RUNNING");
    let processFrame = async () => {
      console.log("RUNNING PROCESS FRAME WITH ONE AS ",ONE);
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }

      if (videoRef.current) {
        if (!ONE) await playCamera();
        const data = await takePicture(videoRef.current, ONE);
        if (ONE) {
          updateContinuousSound(data[0]);
        } else {
          await playRadar(data, PLAY_INTERVAL);
          console.log("played radar");
        }
      } else {
        console.error("Video element not available");
      }
      console.log("SETTING TIMEOUT", ONE ? 0 : 750);
      setTimeout(()=>{
        console.log("RUNNING IT AGAIN!!!",processFrame);
        processFrame();
      }, ONE ? 0 : 750);
    };

    console.log("started", window.ran);
    if (!window.ran) {
      window.ran = true;
      
      console.log("RUNNING PROCESS!",started)
      processFrame();
    }

    return () => {
      processFrame=()=>{}
    };
  }, [ONE, started]);

  const updateContinuousSound = (volume) => {
    if (oscillatorRef.current && gainNodeRef.current) {
      const roundedVolume = Math.round(volume * 4) / 4;
      volume = Math.max(0, Math.min(1, roundedVolume)); // Ensure it's between 0 and 1
      const frequency = volume * 1000; // Calculate target frequency based on volume

      const transitionDuration = 0.05;
      const currentTime = audioCtx.currentTime;

      oscillatorRef.current.frequency.setValueAtTime(
        oscillatorRef.current.frequency.value,
        currentTime
      );

      oscillatorRef.current.frequency.linearRampToValueAtTime(
        frequency,
        currentTime + transitionDuration
      );
    }
  };

  return (
    <div>
      <a
        style={{
          top: "0px",
          left: "0px",
          width: "100%",
          textAlign: "center",
          color: "white",
          position: "fixed",
          transform: "rotate(90deg)",
        }}
      >
        ⬆️
      </a>
      <VideoStream onClick={() => {
       window.ran=false;
       setONE(!ONE);
       setStarted(false);
      }} ref={videoRef} />
      <div  style={{
          top: "0px",
          left: "0px",
          width: "100%",
          textAlign: "center",
          color: "white",
          fontSize:"30px"
        }}>
          {ONE?"Stream":"Radar"}
      </div>
    </div>
  );
}

async function playCamera(volume = 0.5, time = 0.5) {
  return new Promise((resolve) => {
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();
    const pannerNode = audioCtx.createStereoPanner();

    oscillator.type = "sine";
    const frequency = Math.max(100, Math.min(2000, volume * 2000)); // Clamp frequency between 100Hz and 2000Hz
    oscillator.frequency.setValueAtTime(frequency, audioCtx.currentTime);

    gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.5, audioCtx.currentTime + 0.01);
    gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + time);

    pannerNode.pan.value = 0;

    oscillator.connect(pannerNode);
    pannerNode.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.start(audioCtx.currentTime);
    oscillator.stop(audioCtx.currentTime + time);

    oscillator.onended = () => {
      oscillator.disconnect();
      pannerNode.disconnect();
      gainNode.disconnect();
      resolve();
    };
  });
}

export default App;
