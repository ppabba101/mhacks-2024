// Function to Play Radar with Panning and Volume Transitions
export function playRadar(volumes, duration) {
  // Initialize Audio Context
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  const audioCtx = new AudioContext();

  // Create Gain Node for Volume Control
  const gainNode = audioCtx.createGain();
  gainNode.gain.value = 1; // Set initial volume to maximum

  // Create Oscillator
  const oscillator = audioCtx.createOscillator();
  oscillator.type = "sine";
  oscillator.frequency.setValueAtTime(440, audioCtx.currentTime); // A4 Note

  // Create Stereo Panner
  const panner = audioCtx.createStereoPanner();
  panner.pan.value = -1; // Start at left

  // Connect Nodes
  oscillator.connect(panner);
  panner.connect(gainNode);
  gainNode.connect(audioCtx.destination);

  // Start Oscillator
  oscillator.start();

  // Panning from left (-1) to right (1) over the specified duration
  panner.pan.linearRampToValueAtTime(1, audioCtx.currentTime + duration);

  // Calculate time intervals for volume changes
  const numberOfVolumes = volumes.length;
  const interval = duration / (numberOfVolumes - 1);

  // Schedule volume changes
  volumes.forEach((vol, index) => {
    const time = audioCtx.currentTime + index * interval;
    gainNode.gain.linearRampToValueAtTime(vol, time);
  });

  // Ensure the last volume level is set at the end
  gainNode.gain.setValueAtTime(
    volumes[volumes.length - 1],
    audioCtx.currentTime + duration
  );

  // Stop after the specified duration
  oscillator.stop(audioCtx.currentTime + duration);
}

// Example usage:
// const volumes = [0, 0.5, 1, 0.5, 0]; // Example volume array
// const duration = 2; // Example duration in seconds
// playRadar(volumes, duration);
