# WebSocket protocol (`/ws`)

JSON text frames unless noted. Keep server.py and frontend/index.html in sync
with this file.

## Client → Server
| type        | fields                          | meaning |
|-------------|---------------------------------|---------|
| `utterance` | `text`                          | simulated speech from the UI text box |
| `audio`     | binary frame (16 kHz PCM s16le) | one VAD-chunked utterance (browser-mic mode; stretch goal — Phase 4 shipped the backend-owned mic instead: `scribe serve --mic`, no client frame involved) |
| `intent`    | `name: commit\|undo\|scratch\|mute\|wake` | UI button pressed (voice spotters are server-side); `mute`/`wake` toggle the mic gate |
| `edit`      | `action: <Action JSON>`         | keyboard-sourced edit (MathQuill save, delete button) |
| `resolve`   | `pending_id, verdict: commit\|scratch` | resolve a pending proposal |
| `mic`       | `action: list\|select\|test_start\|test_stop, device?: int` | mic device selection + level tester (only when server runs `--mic`; otherwise `error`) |
| `edit_preview` | `target_id, latex \| null`   | live editor keystrokes (client-throttled ~120 ms); `null` = editing ended without save → revert |

## Server → Client
| type         | fields                                   | meaning |
|--------------|------------------------------------------|---------|
| `status`     | `state: listening\|heard\|thinking\|idle\|muted`, `detail?` | drives UI state strip; in mic mode the server starts `muted` (wake-on-wake-word) — "hey Jarvis", the UI unmute button, or `intent:wake` flips it to `listening` (`--start-unmuted` skips the gate) |
| `transcript` | `text, interim: bool`                    | ASR transcript (interim = show greyed) |
| `proposal`   | `pending_id, action: <Action JSON>, transcript?, segments?` | engine proposed an edit → UI shows pending strip; re-sent with the SAME `pending_id` (and growing `segments` count) as dictation extends a pending equation |
| `applied`    | `action, assigned_id?, doc_context`      | DocState mutated (after commit / keyboard edit / undo) |
| `reply`      | `text` / `question, candidates`          | text_reply or clarify from engine |
| `error`      | `message, action?`                       | e.g. unknown target id |
| `mics`       | `devices: [{index, name, default}], current` | input device list; sent on connect in mic mode, re-broadcast after `select` |
| `miclevel`   | `rms, prob`                              | ~10 Hz while the mic tester is on (`prob` = Silero speech probability) |
| `edit_preview` | `target_id, latex \| null`             | relay of a client's live edit keystrokes — transient view-only state: never touches DocState, history, or the datalogger; `applied` (on save) supersedes any preview |

## Flow
1. Utterance arrives (text or audio) → `status:thinking` → engine →
   `proposal` (document Actions) or `reply` (text_reply/clarify).
2. **Composition:** while an `append_math` proposal is pending, further
   dictation EXTENDS it — the server joins the accumulated transcripts,
   re-runs the engine's text path on the whole equation, and re-broadcasts
   `proposal` with the same `pending_id` and an incremented `segments`.
   Commit is explicit: `intent:commit` / voice spotter / typed "commit" →
   server applies to DocState → `applied` + datalogger verdict `committed`
   (one training triple: joined transcript + concatenated audio).
   A command utterance (replace/delete/…) still auto-commits the pending
   composition before being proposed itself.
   `scratch` pops the LAST segment (re-corrects the remainder, re-broadcasts
   the proposal); with one segment left it discards the pending → verdict
   `scratched` (popped segments are logged `scratched` individually).
3. Keyboard `edit` frames apply immediately (verdict `committed`,
   source `keyboard`); if they modify a line committed < N s ago, logger marks
   the earlier record `edited_after` with the corrected LaTeX as gold.
4. **Mute gate (mic mode):** while `muted`, VAD utterances are dropped
   server-side (dictation never reaches the engine) but ALL wake-word
   spotters keep running — "hey Jarvis" fires `wake` (unmute), and
   commit/undo/scratch voice keywords stay active by design. Typed
   `utterance` frames are unaffected. `mute`/`wake` are app-layer intents,
   never engine Actions.
5. `undo` pops DocState history → `applied` with full new `doc_context`
   (client re-renders whole doc from `doc_context` on every `applied` — the
   server is the single source of truth).
