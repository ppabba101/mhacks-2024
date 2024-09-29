async function sleep(time) {
  return new Promise((a) => {
    setTimeout(a, time);
  });
}

export async function playRadar(volumes, duration) {
  // Initialize Audio Context
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioContext();

  const gainNode = audioCtx.createGain();
  const oscillator = audioCtx.createOscillator();
  const panner = audioCtx.createStereoPanner();

  oscillator.connect(panner);
  panner.connect(gainNode);
  gainNode.connect(audioCtx.destination);

  console.log("playing first ding");
  await playDing(audioCtx, "left");
  console.log("JUST FINIESHED THE DING");

  
  await sleep(500);
  // gainNode.gain.value = 1; // Set initial volume to maximum
  console.log("after sleeping");
  // Rest of the function remains the same...
  // Panning from left (-1) to right (1) over the specified duration
  gainNode.gain.value = 0;
  oscillator.start();
  oscillator.type = "sine";
  oscillator.frequency.setValueAtTime(440, audioCtx.currentTime); // A4 Note
  panner.pan.value = -1; // Start at left
  panner.pan.linearRampToValueAtTime(1, audioCtx.currentTime + duration);

  // Calculate time intervals for volume changes
  const numberOfVolumes = volumes.length;
  const interval = duration / (numberOfVolumes - 1);

  // Schedule volume changes
  volumes.forEach((vol, index) => {
    const time = audioCtx.currentTime + index * interval;
    gainNode.gain.setTargetAtTime(vol, time, 0.1);
  });

  // Ensure the last volume level is set at the end
  gainNode.gain.setTargetAtTime(
    volumes[volumes.length - 1],
    audioCtx.currentTime + duration,
    0.1
  );

  // Stop the radar sound after the specified duration
  oscillator.stop(audioCtx.currentTime + duration);

  await sleep(duration * 1000+500);

  // Play a "ding" sound after the radar sound (panned to the right)
  console.log("playing second ding");
  await playDing(audioCtx, "right");
}

// Function to play the "ding" sound with panning
async function playDing(audioCtx, direction) {
  return new Promise((resolve) => {
    const oscillator = audioCtx.createOscillator();
    const dingGain = audioCtx.createGain();
    const dingPanner = audioCtx.createStereoPanner();

    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 Note

    dingGain.gain.setValueAtTime(0, audioCtx.currentTime);
    dingGain.gain.linearRampToValueAtTime(0.5, audioCtx.currentTime + 0.01);
    dingGain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.5);

    dingPanner.pan.value = direction === "left" ? -1 : 1;

    oscillator.connect(dingPanner);
    dingPanner.connect(dingGain);
    dingGain.connect(audioCtx.destination);

    oscillator.start(audioCtx.currentTime);
    oscillator.stop(audioCtx.currentTime + 0.5);

    oscillator.onended = () => {
      oscillator.disconnect();
      dingPanner.disconnect();
      dingGain.disconnect();
      resolve();
    };
  });
}
