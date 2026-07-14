"""Silero VAD chunker over a 16 kHz mono mic stream (sounddevice).
Emits utterance wavs (0.5–10 s, padded). Implement in PLAN.md Phase 4.
Runs in its own thread; pushes (float32 pcm, duration) onto an asyncio queue
consumed by server.py."""
# TODO(Phase 4)
