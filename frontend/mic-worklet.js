// AudioWorklet: capture-rate float32 -> 16 kHz s16le, posted to the main
// thread as 512-sample (1024-byte) ArrayBuffers — the frame size the
// server-side VAD pipeline expects (PROTOCOL.md `audio`).
//
// `sampleRate` is the worklet-global AudioContext rate. Chrome honours the
// {sampleRate:16000} hint so the ratio is usually 1; browsers that ignore it
// (or a device pinned at 48 kHz) go through the same linear-interpolation
// path, so behavior is uniform everywhere.
const TARGET_RATE = 16000;
const FRAME_SAMPLES = 512;

class Pcm16Downsampler extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.carry = new Float32Array(0);   // unconsumed input tail
    this.pos = 0;                       // fractional read position in carry
    this.out = new Int16Array(FRAME_SAMPLES);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || !ch.length) return true;
    const buf = new Float32Array(this.carry.length + ch.length);
    buf.set(this.carry);
    buf.set(ch, this.carry.length);

    let pos = this.pos;
    while (pos + 1 < buf.length) {
      const i = Math.floor(pos), frac = pos - i;
      const s = buf[i] * (1 - frac) + buf[i + 1] * frac;
      const clamped = Math.max(-1, Math.min(1, s));
      this.out[this.n++] = Math.round(clamped * 32767);
      if (this.n === FRAME_SAMPLES) {
        const frame = this.out.slice(0);          // fresh copy to transfer
        this.port.postMessage(frame.buffer, [frame.buffer]);
        this.n = 0;
      }
      pos += this.ratio;
    }
    const keep = Math.floor(pos);
    this.carry = buf.slice(keep);
    this.pos = pos - keep;
    return true;
  }
}

registerProcessor('pcm16-downsampler', Pcm16Downsampler);
