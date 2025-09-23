class TTSPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferQueue = [];
    this.readOffset = 0;
    this.samplesRemaining = 0;
    this.isPlaying = false;

    // Listen for incoming messages
    this.port.onmessage = (event) => {
      // Check if this is a control message (object with a "type" property).
      if (
        event.data &&
        typeof event.data === "object" &&
        event.data.type === "clear"
      ) {
        // Clear the TTS buffer and reset playback state.
        console.log("TTS Worklet: Clearing buffer");
        this.bufferQueue = [];
        this.readOffset = 0;
        this.samplesRemaining = 0;
        this.isPlaying = false;
        return;
      }

      // Otherwise assume it's a PCM chunk (e.g., an Int16Array)
      if (event.data instanceof Int16Array) {
        console.log(
          `TTS Worklet: Received ${event.data.length} samples, total queued: ${
            this.samplesRemaining + event.data.length
          }`
        );
        this.bufferQueue.push(event.data);
        this.samplesRemaining += event.data.length;
      } else {
        console.warn(
          "TTS Worklet: Received unexpected data type:",
          typeof event.data
        );
      }
    };
  }

  process(inputs, outputs) {
    const outputChannel = outputs[0][0];

    if (this.samplesRemaining === 0) {
      outputChannel.fill(0);
      if (this.isPlaying) {
        this.isPlaying = false;
        this.port.postMessage({ type: "ttsPlaybackStopped" });
      }
      return true;
    }

    if (!this.isPlaying) {
      this.isPlaying = true;
      console.log("TTS Worklet: Starting playback");
      this.port.postMessage({ type: "ttsPlaybackStarted" });
    }

    let outIdx = 0;
    while (outIdx < outputChannel.length && this.bufferQueue.length > 0) {
      const currentBuffer = this.bufferQueue[0];
      const sampleValue = currentBuffer[this.readOffset] / 32768;
      outputChannel[outIdx++] = sampleValue;

      this.readOffset++;
      this.samplesRemaining--;

      if (this.readOffset >= currentBuffer.length) {
        this.bufferQueue.shift();
        this.readOffset = 0;
      }
    }

    while (outIdx < outputChannel.length) {
      outputChannel[outIdx++] = 0;
    }

    return true;
  }
}

registerProcessor("tts-playback-processor", TTSPlaybackProcessor);
