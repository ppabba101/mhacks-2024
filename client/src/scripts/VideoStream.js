import React, { useRef, useEffect, forwardRef } from "react";

const VideoStream = forwardRef((props, ref) => {
  const videoRef = useRef(null);

  useEffect(() => {
    const startStream = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { exact: "environment" },
          },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing the back camera:", err);
      }
    };

    startStream();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // Use the forwarded ref or the internal ref
  const combinedRef = (node) => {
    videoRef.current = node;
    if (typeof ref === "function") {
      ref(node);
    } else if (ref) {
      ref.current = node;
    }
  };

  return <video ref={combinedRef} autoPlay playsInline />;
});

export default VideoStream;
