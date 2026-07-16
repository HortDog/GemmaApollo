"""FastAPI app: serves frontend/ and the /ws endpoint per PROTOCOL.md.
Server owns DocState (single source of truth) and the SessionLogger.
Phase 2: full loop with MockEngine. Phase 4: backend-owned mic — VAD-chunked
utterances feed the same pipeline as typed ones; all server frames broadcast
to every connected client (single-user app, possibly several tabs)."""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter

from .datalogger import SessionLogger
from .docstate import DocState
from .engine.base import Action, Clarify, TextReply
from .engine.mock_engine import MockEngine

ACTION_ADAPTER = TypeAdapter(Action)

def make_engine(name: str):
    if name == "mock":
        return MockEngine()
    if name == "s2l":
        from .engine.s2l_engine import S2LEngine
        return S2LEngine()
    if name == "gemma":
        from .engine.gemma_engine import GemmaEngine
        return GemmaEngine()
    raise ValueError(name)

def build_app(engine_name: str = "mock", mic: bool = False,
              wakeword_models: dict[str, str] | None = None,
              mic_device: int | None = None,
              start_unmuted: bool = False) -> FastAPI:
    """wakeword_models: {model_path_or_name: intent} for the Phase 5 spotters
    (requires mic=True). None -> auto-load DEFAULT_MODELS paths that exist.
    mic_device: input device index (None = system default).
    start_unmuted: skip the wake-on-wake-word gate — mic mode normally starts
    muted until "hey Jarvis" / the UI unmute button fires a `wake` intent."""
    app = FastAPI(title="GemmaApollo Scribe")
    engine = make_engine(engine_name)
    doc = DocState()
    logger = SessionLogger()
    clients: set[WebSocket] = set()
    pending: dict = {}  # pending_id -> {"action", "transcript", "audio_wav"}
    counter = {"n": 0}
    # Mic supervisor hooks, populated by start_mic (dispatch runs earlier in
    # the file; tests stub these). Keys: list / select / set_testing.
    micctl: dict = {}
    app.state.micctl = micctl
    # Mic gate (wake-on-wake-word): while muted, VAD/spotters keep running but
    # dictation never reaches the engine. `wake`/`mute` intents flip it.
    mute = {"on": mic and not start_unmuted}

    def mic_state() -> str:
        return ("muted" if mute["on"] else "listening") if mic else "idle"

    # ------------------------------------------------------------ transport
    async def send(ws: WebSocket, obj: dict):
        await ws.send_text(json.dumps(obj))

    async def broadcast(obj: dict):
        dead = []
        for ws in clients:
            try:
                await send(ws, obj)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    # ------------------------------------------------------------ doc ops
    async def apply_and_broadcast(action: Action, verdict: str,
                                  transcript=None, audio_wav: bytes | None = None,
                                  source="voice"):
        # Training triples are (doc context -> action): log the context the
        # action was issued against, i.e. BEFORE the mutation.
        ctx_before = doc.render_context()
        try:
            assigned = doc.apply(action)
        except KeyError as e:
            await broadcast({"type": "error", "message": str(e),
                             "action": action.model_dump()})
            logger.log(audio_bytes=audio_wav, doc_context=ctx_before,
                       transcript=transcript, engine_action=action.model_dump(),
                       final_action=None, verdict="discarded",
                       engine=engine.name, latency_ms={})
            return
        logger.log(audio_bytes=audio_wav, doc_context=ctx_before,
                   transcript=transcript, engine_action=action.model_dump(),
                   final_action=action.model_dump(), verdict=verdict,
                   engine=engine.name, latency_ms={})
        await broadcast({"type": "applied", "action": action.model_dump(),
                         "assigned_id": assigned,
                         "doc_context": doc.render_context()})

    # ---------------------------------------------------- pending composition
    # A pending append_math proposal is a COMPOSITION: further dictation
    # extends it (transcripts re-corrected as one equation) until the user
    # commits or scratches. This is the app-layer system that lets
    # non-agentic engines (S2L) build long equations without edit-op tools.
    # segments: [{"text": str, "audio": np.ndarray|None}]

    def composed_wav(p) -> bytes | None:
        chunks = [s["audio"] for s in p["segments"] if s.get("audio") is not None]
        if not chunks:
            return None
        import numpy as np

        from .audio.vad import wav_bytes
        return wav_bytes(np.concatenate(chunks))

    def joined_text(p) -> str:
        return " ".join(s["text"] for s in p["segments"] if s["text"])

    async def broadcast_proposal(pid: str):
        p = pending[pid]
        await broadcast({"type": "proposal", "pending_id": pid,
                         "action": p["action"].model_dump(),
                         "transcript": p["transcript"],
                         "segments": len(p["segments"])})

    async def remerge(p):
        """Re-run the engine's text path on the joined transcript so the
        post-corrector sees the whole equation, not glued fragments."""
        joined = joined_text(p)
        merged = await asyncio.to_thread(engine.process_text, joined,
                                         doc.render_context())
        if merged.action.action == "append_math":
            p["action"] = merged.action
        # else: joined text suddenly parses as a command — keep previous latex
        p["transcript"] = joined

    async def resolve_pending(pending_id: str, verdict: str):
        """Commit or scratch a pending proposal. Shared by `resolve` frames and
        `commit`/`scratch` intents (buttons now, voice spotters in Phase 5).
        Scratch pops the last composition segment first; the whole pending is
        discarded only when a single segment remains."""
        p = pending.get(pending_id)
        if not p:
            return
        if verdict == "commit":
            pending.pop(pending_id)
            await apply_and_broadcast(p["action"], "committed",
                                      p["transcript"], composed_wav(p))
            return
        if len(p["segments"]) > 1:
            popped = p["segments"].pop()
            audio_wav = None
            if popped.get("audio") is not None:
                from .audio.vad import wav_bytes
                audio_wav = wav_bytes(popped["audio"])
            logger.log(audio_bytes=audio_wav, doc_context=doc.render_context(),
                       transcript=popped["text"],
                       engine_action=p["action"].model_dump(),
                       final_action=None, verdict="scratched",
                       engine=engine.name, latency_ms={})
            await remerge(p)
            await broadcast_proposal(pending_id)
        else:
            pending.pop(pending_id)
            logger.log(audio_bytes=composed_wav(p),
                       doc_context=doc.render_context(),
                       transcript=p["transcript"],
                       engine_action=p["action"].model_dump(),
                       final_action=None, verdict="scratched",
                       engine=engine.name, latency_ms={})
            await broadcast({"type": "status", "state": "idle",
                             "detail": "scratched"})

    # ------------------------------------------------------------ utterances
    def engine_context() -> str:
        """Doc context for the engine. A pending append is shown under the id
        it WILL get on commit, so voice commands can target it (auto-commit
        assigns exactly that id before the command applies)."""
        ctx = doc.render_context()
        for p in pending.values():
            if p["action"].action == "append_math":
                entry = f"[{doc.peek_next_id()}] {p['action'].latex}"
                ctx = f"{ctx}  {entry}" if ctx else entry
        return ctx

    async def process_utterance(text: str | None = None, audio=None):
        """Shared pipeline for typed utterances (ws text box) and mic
        utterances (VAD chunker). Engine inference runs off the event loop."""
        await broadcast({"type": "status", "state": "thinking"})
        if audio is not None:
            res = await asyncio.to_thread(engine.process, audio,
                                          engine_context())
        else:
            res = await asyncio.to_thread(engine.process_text, text,
                                          engine_context())
        if res.transcript:
            await broadcast({"type": "transcript", "text": res.transcript,
                             "interim": False})
        a = res.action
        segment = {"text": res.transcript or text or "", "audio": audio}
        composing = next((pid for pid, p in pending.items()
                          if p["action"].action == "append_math"), None)
        if isinstance(a, (TextReply, Clarify)):
            await broadcast({"type": "reply", **a.model_dump()})
        elif a.action == "append_math" and composing is not None:
            # extend the pending equation instead of starting a new line
            p = pending[composing]
            p["segments"].append(segment)
            await remerge(p)
            await broadcast_proposal(composing)
        else:
            # commands auto-commit whatever is pending (they end composition);
            # a fresh append starts a new composition after the same flush
            for pid in list(pending):
                await resolve_pending(pid, "commit")
            counter["n"] += 1
            pid = f"p{counter['n']}"
            pending[pid] = {"action": a, "transcript": segment["text"],
                            "segments": [segment]}
            await broadcast_proposal(pid)
        await broadcast({"type": "status", "state": mic_state()})

    # ------------------------------------------------------------ intents
    async def handle_intent(name: str, source: str = "ui"):
        """App-layer intents — UI buttons and wake-word spotters share this
        exact path. Never engine Actions."""
        if name in ("wake", "mute"):
            # Flip the gate BEFORE the status broadcast so it reports the
            # new state ("hey Jarvis" -> listening, UI mute -> muted).
            mute["on"] = name == "mute"
        if source == "wakeword":
            # PLAN.md Phase 5: log every spotter fire (false-positive audit).
            logger.log(audio_bytes=None, doc_context=doc.render_context(),
                       transcript=None,
                       engine_action={"intent": name, "source": source},
                       final_action=None, verdict="app_intent",
                       engine=engine.name, latency_ms={})
            await broadcast({"type": "status", "state": mic_state(),
                             "detail": f"wakeword: {name}"})
        elif name in ("wake", "mute"):
            await broadcast({"type": "status", "state": mic_state(),
                             "detail": "muted" if mute["on"] else "unmuted"})
        if name == "undo":
            if doc.undo():
                await broadcast({"type": "applied",
                                 "action": {"action": "undo"},
                                 "doc_context": doc.render_context()})
            else:
                await broadcast({"type": "status", "state": "idle",
                                 "detail": "nothing to undo"})
        elif name in ("commit", "scratch") and pending:
            # Resolve the oldest outstanding proposal.
            pid = next(iter(pending))
            await resolve_pending(pid, "commit" if name == "commit" else "scratch")

    # ------------------------------------------------------------ dispatch
    async def dispatch(ws: WebSocket, msg: dict):
        """Handle one client frame (see PROTOCOL.md)."""
        t = msg.get("type")

        if t == "utterance":
            await process_utterance(text=msg["text"])

        elif t == "resolve":
            await resolve_pending(msg["pending_id"], msg["verdict"])

        elif t == "intent":
            await handle_intent(msg["name"], source="ui")

        elif t == "edit":
            a = ACTION_ADAPTER.validate_python(msg["action"])
            await apply_and_broadcast(a, "committed", source="keyboard")
            # TODO(Phase 6): edited_after gold-label linkage.

        elif t == "edit_preview":
            # Transient editor keystrokes: relay only. Never touches DocState,
            # history, or the datalogger — `applied` on save is the real
            # mutation. latex=None means the edit ended without saving.
            await broadcast({"type": "edit_preview",
                             "target_id": msg["target_id"],
                             "latex": msg.get("latex")})

        elif t == "mic":
            # Device selection + tester (see PROTOCOL.md). No-op frames when
            # the server runs without --mic.
            if not micctl:
                await send(ws, {"type": "error",
                                "message": "mic mode is off (start with --mic)"})
                return
            action = msg.get("action")
            if action == "list":
                await broadcast(await micctl["list"]())
            elif action == "select":
                await micctl["select"](int(msg["device"]))
            elif action == "test_start":
                micctl["set_testing"](True)
                await broadcast({"type": "status", "state": mic_state(),
                                 "detail": "mic test on"})
            elif action == "test_stop":
                micctl["set_testing"](False)
                await broadcast({"type": "status", "state": mic_state(),
                                 "detail": "mic test off"})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        await send(ws, {"type": "status", "state": mic_state(),
                        "detail": f"engine={engine.name}"})
        if micctl:
            try:
                await send(ws, await micctl["list"]())
            except Exception:
                pass  # device enumeration failure must not block the session
        try:
            while True:
                raw = await ws.receive_text()
                # Contain per-frame failures (bad JSON, missing keys, invalid
                # Action payloads): report an error frame, keep the socket up.
                try:
                    await dispatch(ws, json.loads(raw))
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    await send(ws, {"type": "error",
                                    "message": f"{type(e).__name__}: {e}"})
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    # ------------------------------------------------------------ mic (Phase 4)
    if mic:
        @app.on_event("startup")
        async def start_mic():
            import numpy as np

            from .audio.vad import SileroVAD, UtteranceChunker, list_input_devices, mic_frames

            loop = asyncio.get_running_loop()
            utterances: asyncio.Queue = asyncio.Queue()
            intents: asyncio.Queue = asyncio.Queue()
            levels: asyncio.Queue = asyncio.Queue()
            # Shared across device switches: Silero + oww models load once.
            vad = SileroVAD()
            state = {"device": mic_device, "stop": None, "thread": None,
                     "testing": False, "scorer": None}

            def make_scorer():
                """Phase 5 wake-word scorer; None if no models available."""
                from pathlib import Path as P

                from .audio.wakewords import DEFAULT_MODELS, OWWScorer
                mapping = wakeword_models
                if mapping is None:
                    # Custom models only when trained; bare pretrained names
                    # (no path separator, e.g. hey_jarvis_v0.1) always pass —
                    # OWWScorer resolves/downloads them lazily.
                    mapping = {p: intent for intent, p in DEFAULT_MODELS.items()
                               if P(p).exists() or "/" not in p}
                if not mapping:
                    print("wakewords: no models found — spotters disabled "
                          "(train with tools/wakewords/)", flush=True)
                    return None
                print(f"wakewords: spotting {list(mapping.values())}", flush=True)
                return OWWScorer(mapping)

            state["scorer"] = make_scorer()

            def capture(device, stop):  # daemon thread: mic -> chunker|spotter|levels
                from .audio.wakewords import IntentSpotter
                try:
                    vad.reset()  # don't carry VAD state across devices
                    chunker = UtteranceChunker(is_speech=vad)
                    spotter = (IntentSpotter(score=state["scorer"])
                               if state["scorer"] else None)
                    was_muted = mute["on"]
                    for i, frame in enumerate(mic_frames(device, stop=stop)):
                        prob = vad(frame)
                        # Spotters see every frame, in parallel with the VAD —
                        # a hit bypasses the engine entirely (app intent).
                        # They run while muted too: "hey Jarvis" must wake,
                        # and commit/undo/scratch stay active by design.
                        if spotter is not None:
                            hit = spotter.feed(frame)
                            if hit:
                                loop.call_soon_threadsafe(intents.put_nowait, hit)
                        # Mute gates dictation only: while muted the chunker is
                        # starved, and a half-captured utterance is dropped.
                        if mute["on"] != was_muted:
                            was_muted = mute["on"]
                            if was_muted:
                                chunker.reset()
                        if not mute["on"]:
                            u = chunker.feed(frame, prob=prob)
                            if u is not None:
                                loop.call_soon_threadsafe(utterances.put_nowait, u)
                        if state["testing"] and i % 3 == 0:  # ~10 Hz
                            rms = float(np.sqrt(float((frame ** 2).mean())))
                            loop.call_soon_threadsafe(
                                levels.put_nowait, {"rms": rms, "prob": prob})
                except Exception as e:  # no input device, driver error, ...
                    msg = f"mic capture failed: {type(e).__name__}: {e}"
                    print(msg, flush=True)
                    asyncio.run_coroutine_threadsafe(
                        broadcast({"type": "status", "state": "idle",
                                   "detail": msg}), loop)

            def start_capture(device):
                stop = threading.Event()
                t = threading.Thread(target=capture, args=(device, stop),
                                     daemon=True, name="mic")
                state.update(device=device, stop=stop, thread=t)
                t.start()

            def stop_capture():
                if state["stop"] is not None:
                    state["stop"].set()
                    state["thread"].join(timeout=2)

            # ---- micctl: hooks used by the ws dispatch (PROTOCOL.md `mic`) ----
            async def mics_frame():
                devices = await asyncio.to_thread(list_input_devices)
                return {"type": "mics", "devices": devices,
                        "current": state["device"]}

            async def select(device: int):
                def probe():
                    import sounddevice as sd
                    sd.check_input_settings(device=device, samplerate=16000,
                                            channels=1)
                try:
                    await asyncio.to_thread(probe)
                except Exception as e:
                    await broadcast({"type": "error",
                                     "message": f"mic device {device}: {e}"})
                    return
                await asyncio.to_thread(stop_capture)
                start_capture(device)
                await broadcast(await mics_frame())

            micctl["list"] = mics_frame
            micctl["select"] = select
            micctl["set_testing"] = lambda on: state.update(testing=on)

            async def consume_utterances():
                while True:
                    u = await utterances.get()
                    if mute["on"]:
                        continue  # muted while this one was already queued
                    await broadcast({"type": "status", "state": "heard",
                                     "detail": f"utterance {len(u)/16000:.1f}s"})
                    await process_utterance(audio=u)

            async def consume_intents():
                # Separate task so commit/undo work while the engine is busy.
                while True:
                    name = await intents.get()
                    await handle_intent(name, source="wakeword")

            async def consume_levels():
                while True:
                    lvl = await levels.get()
                    await broadcast({"type": "miclevel", **lvl})

            start_capture(mic_device)
            asyncio.create_task(consume_utterances())
            asyncio.create_task(consume_intents())
            asyncio.create_task(consume_levels())

    app.mount("/", StaticFiles(directory=Path(__file__).resolve()
              .parents[2] / "frontend", html=True), name="frontend")
    return app
