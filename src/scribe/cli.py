"""CLI entry: scribe serve | bench | eval | export."""
import argparse

def main():
    p = argparse.ArgumentParser(prog="scribe")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve"); s.add_argument("--engine", default="mock",
        choices=["mock", "s2l", "gemma"]); s.add_argument("--port", type=int, default=8017)
    sub.add_parser("bench").add_argument("--engine", default="s2l")
    sub.add_parser("eval").add_argument("--set", default="realvoice")
    sub.add_parser("export")
    args = p.parse_args()
    if args.cmd == "serve":
        import uvicorn
        from .server import build_app
        uvicorn.run(build_app(engine_name=args.engine), host="127.0.0.1", port=args.port)
    else:
        raise SystemExit(f"'{args.cmd}' not implemented yet — see PLAN.md")
