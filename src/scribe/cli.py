"""CLI entry: scribe serve | infer-serve | bench | mics | mic-test | eval | export."""
import argparse
import os

DEFAULT_INFER_URL = os.environ.get("SCRIBE_INFER_URL", "http://127.0.0.1:8018")


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
        choices=["mock", "s2l", "gemma", "remote"])
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8017)
    s.add_argument("--infer-url", default=DEFAULT_INFER_URL,
                   help="inference server URL for --engine remote "
                        "(env SCRIBE_INFER_URL)")
    s.add_argument("--mic", action="store_true",
                   help="backend owns the microphone: VAD-chunked utterances feed the engine")
    s.add_argument("--mic-device", type=int, default=None,
                   help="input device index (see `scribe mics`); default: system default")
    s.add_argument("--wakeword-model", action="append", default=None,
        metavar="PATH=INTENT",
        help="spotter model mapping, repeatable (e.g. models/commit.onnx=commit); "
             "default: auto-load models/wakewords/* if present")
    i = sub.add_parser("infer-serve",
        help="standalone inference server hosting the heavy engine (INFERENCE.md)")
    i.add_argument("--engine", default="s2l", choices=["mock", "s2l", "gemma"])
    i.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 exposes the server to your LAN — no auth, "
                        "trusted networks only")
    i.add_argument("--port", type=int, default=8018)
    i.add_argument("--device", default="auto",
                   help="auto | cuda | cpu (auto: cuda if available)")
    i.add_argument("--asr-model", default=None,
                   help="override whisper model: large-v3 | large-v3-turbo | distil-large-v3")
    i.add_argument("--preload", action="store_true",
                   help="load models at startup (readiness via /readyz) instead "
                        "of on the first request")
    b = sub.add_parser("bench"); b.add_argument("--engine", default="s2l",
        choices=["mock", "s2l", "remote"]); b.add_argument("--asr-model", default=None,
        help="override whisper model: large-v3 | large-v3-turbo | distil-large-v3")
    b.add_argument("--infer-url", default=DEFAULT_INFER_URL,
                   help="inference server URL for --engine remote")
    sub.add_parser("mics", help="list audio input devices")
    mt = sub.add_parser("mic-test", help="live console level/VAD tester")
    mt.add_argument("--device", type=int, default=None)
    mt.add_argument("--seconds", type=float, default=15)
    sub.add_parser("eval").add_argument("--set", default="realvoice")
    sub.add_parser("export")
    args = p.parse_args()
    if args.cmd == "serve":
        import uvicorn

        from .server import build_app
        ww = None
        if args.wakeword_model:
            ww = dict(spec.split("=", 1) for spec in args.wakeword_model)
        ekw = {"url": args.infer_url} if args.engine == "remote" else None
        uvicorn.run(build_app(engine_name=args.engine, mic=args.mic,
                              wakeword_models=ww, mic_device=args.mic_device,
                              engine_kwargs=ekw),
                    host=args.host, port=args.port)
    elif args.cmd == "infer-serve":
        import uvicorn

        from .infer_server import build_infer_app
        ekw = {}
        if args.engine != "mock":
            ekw["device"] = args.device
        if args.asr_model and args.engine == "s2l":
            ekw["asr_model"] = args.asr_model
        uvicorn.run(build_infer_app(engine_name=args.engine,
                                    preload=args.preload, engine_kwargs=ekw),
                    host=args.host, port=args.port)
    elif args.cmd == "bench":
        from .bench import print_summary, run_bench
        ekw = {"url": args.infer_url} if args.engine == "remote" else None
        print_summary(run_bench(args.engine, asr_model=args.asr_model,
                                engine_kwargs=ekw))
    elif args.cmd == "mics":
        from .audio.vad import list_input_devices
        for d in list_input_devices():
            star = "*" if d["default"] else " "
            print(f"{star} [{d['index']:>2}] {d['name']}")
    elif args.cmd == "mic-test":
        _mic_test(args.device, args.seconds)
    else:
        raise SystemExit(f"'{args.cmd}' not implemented yet — see PLAN.md")
