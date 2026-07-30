"""CLI entry: scribe serve | model-server | bench | mics | mic-test | eval | export."""
import argparse


def _watch_parent_stdin():
    """Orphan guard for the Electron shell (SCRIBE_PARENT_WATCH=1): the
    parent holds our stdin pipe, so EOF means it died or wants us gone —
    exit instead of lingering as a headless server."""
    import os
    import sys
    import threading

    def watch():
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=watch, daemon=True, name="parentwatch").start()


def _mic_test(device: int | None, seconds: float):
    """Standalone console mic tester: live RMS bar + Silero prob + chunker
    events. No server involved — exercises the exact capture stack."""
    import sys
    import threading
    import time

    import numpy as np

    from .audio.vad import SAMPLE_RATE, SileroVAD, UtteranceChunker, mic_frames

    vad = SileroVAD()
    chunker = UtteranceChunker(is_speech=vad)
    stop = threading.Event()
    t_end = time.monotonic() + seconds
    utt_count = 0
    print(f"mic test: device={'default' if device is None else device}, "
          f"{seconds:.0f}s — speak to see the bar move (utterances are chunked)")
    try:
        for frame in mic_frames(device, stop=stop):
            prob = vad(frame)
            u = chunker.feed(frame, prob=prob)
            if u is not None:
                utt_count += 1
                print(f"\n  utterance captured: {len(u)/SAMPLE_RATE:.1f}s")
            rms = float(np.sqrt(float((frame ** 2).mean())))
            bar = "#" * min(40, int(rms * 400))
            mark = "V" if prob >= 0.5 else " "
            sys.stdout.write(f"\r[{mark}] {bar:<40} rms={rms:.3f} vad={prob:.2f}  ")
            sys.stdout.flush()
            if time.monotonic() >= t_end:
                stop.set()
    except KeyboardInterrupt:
        stop.set()
    print(f"\ndone: {utt_count} utterance(s) captured")


def main():
    p = argparse.ArgumentParser(prog="scribe")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--engine", default="mock",
        choices=["mock", "s2l", "gemma", "remote"]); s.add_argument("--port", type=int, default=8017)
    s.add_argument("--host", default="127.0.0.1",
                   help="bind address (default loopback; the Electron shell "
                        "and tailscale serve both proxy to loopback)")
    s.add_argument("--model-server-url", default="http://127.0.0.1:8018",
                   help="inference service for --engine remote; datalogger "
                        "rows also mirror to its central /log store")
    s.add_argument("--mic", action="store_true",
                   help="backend owns the microphone: VAD-chunked utterances feed the engine")
    s.add_argument("--mic-device", type=int, default=None,
                   help="input device index (see `scribe mics`); default: system default")
    s.add_argument("--start-unmuted", action="store_true",
                   help="mic mode starts listening immediately instead of "
                        "muted-until-'hey Jarvis' (the wake intent)")
    s.add_argument("--wakeword-model", action="append", default=None,
        metavar="PATH=INTENT",
        help="spotter model mapping, repeatable (e.g. models/commit.onnx=commit); "
             "default: auto-load models/wakewords/* if present")
    m = sub.add_parser("model-server",
        help="Engine-over-HTTP inference service + central /log dataset store")
    m.add_argument("--engine", default="s2l", choices=["mock", "s2l", "gemma"])
    m.add_argument("--host", default="127.0.0.1",
                   help="bind address (Tailscale 100.x IP for cross-machine "
                        "use; 0.0.0.0 inside Docker)")
    m.add_argument("--port", type=int, default=8018)
    m.add_argument("--preload", action="store_true",
                   help="load models before serving traffic (otherwise the "
                        "first utterance waits out the lazy load)")
    b = sub.add_parser("bench"); b.add_argument("--engine", default="s2l",
        choices=["mock", "s2l"]); b.add_argument("--asr-model", default=None,
        help="override whisper model: large-v3 | large-v3-turbo | distil-large-v3")
    sub.add_parser("mics", help="list audio input devices")
    mt = sub.add_parser("mic-test", help="live console level/VAD tester")
    mt.add_argument("--device", type=int, default=None)
    mt.add_argument("--seconds", type=float, default=15)
    sub.add_parser("eval").add_argument("--set", default="realvoice")
    sub.add_parser("export")
    args = p.parse_args()
    if args.cmd == "serve":
        import os

        import uvicorn

        from .server import build_app
        if os.environ.get("SCRIBE_PARENT_WATCH"):
            _watch_parent_stdin()
        ww = None
        if args.wakeword_model:
            ww = dict(spec.split("=", 1) for spec in args.wakeword_model)
        uvicorn.run(build_app(engine_name=args.engine, mic=args.mic,
                              wakeword_models=ww, mic_device=args.mic_device,
                              start_unmuted=args.start_unmuted,
                              model_server_url=(args.model_server_url
                                                if args.engine == "remote"
                                                else None)),
                    host=args.host, port=args.port)
    elif args.cmd == "model-server":
        import uvicorn

        from .model_server import build_model_app
        app = build_model_app(args.engine)
        if args.preload:
            print(f"preloading engine '{args.engine}'...", flush=True)
            app.state.warmup()
            print("engine ready", flush=True)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.cmd == "bench":
        from .bench import print_summary, run_bench
        print_summary(run_bench(args.engine, asr_model=args.asr_model))
    elif args.cmd == "mics":
        from .audio.vad import list_input_devices
        for d in list_input_devices():
            star = "*" if d["default"] else " "
            print(f"{star} [{d['index']:>2}] {d['name']}")
    elif args.cmd == "mic-test":
        _mic_test(args.device, args.seconds)
    else:
        raise SystemExit(f"'{args.cmd}' not implemented yet — see PLAN.md")
