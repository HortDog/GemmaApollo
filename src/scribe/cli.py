"""CLI entry: scribe serve | bench | eval | export."""
import argparse

def main():
    p = argparse.ArgumentParser(prog="scribe")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--engine", default="mock",
        choices=["mock", "s2l", "gemma"]); s.add_argument("--port", type=int, default=8017)
    b = sub.add_parser("bench"); b.add_argument("--engine", default="s2l",
        choices=["mock", "s2l"]); b.add_argument("--asr-model", default=None,
        help="override whisper model: large-v3 | large-v3-turbo | distil-large-v3")
    sub.add_parser("eval").add_argument("--set", default="realvoice")
    sub.add_parser("export")
    args = p.parse_args()
    if args.cmd == "serve":
        import uvicorn
        from .server import build_app
        uvicorn.run(build_app(engine_name=args.engine), host="127.0.0.1", port=args.port)
    elif args.cmd == "bench":
        from .bench import print_summary, run_bench
        print_summary(run_bench(args.engine, asr_model=args.asr_model))
    else:
        raise SystemExit(f"'{args.cmd}' not implemented yet — see PLAN.md")
