# WebSocket protocol (`/ws`)

JSON text frames unless noted. Keep server.py and frontend/index.html in sync
with this file.

## Client → Server
| type        | fields                          | meaning |
|-------------|---------------------------------|---------|
| `utterance` | `text`                          | simulated speech from the UI text box |
| `audio`     | binary frame (16 kHz PCM s16le) | one VAD-chunked utterance (browser-mic mode; stretch goal — Phase 4 shipped the backend-owned mic instead: `scribe serve --mic`, no client frame involved) |
| `intent`    | `name: commit\|undo\|scratch`   | UI button pressed (voice spotters are server-side) |
| `edit`      | `action: <Action JSON>`         | keyboard-sourced edit (MathQuill save, delete button) |
| `resolve`   | `pending_id, verdict: commit\|scratch` | resolve a pending proposal |

## Server → Client
| type         | fields                                   | meaning |
|--------------|------------------------------------------|---------|
| `status`     | `state: listening\|heard\|thinking\|idle`, `detail?` | drives UI state strip |
| `transcript` | `text, interim: bool`                    | ASR transcript (interim = show greyed) |
| `proposal`   | `pending_id, action: <Action JSON>, transcript?` | engine proposed an edit → UI shows pending strip |
| `applied`    | `action, assigned_id?, doc_context`      | DocState mutated (after commit / keyboard edit / undo) |
| `reply`      | `text` / `question, candidates`          | text_reply or clarify from engine |
| `error`      | `message, action?`                       | e.g. unknown target id |

## Flow
1. Utterance arrives (text or audio) → `status:thinking` → engine →
   `proposal` (document Actions) or `reply` (text_reply/clarify).
2. Proposal sits pending. `intent:commit` / voice spotter / a new dictation
   utterance (auto-commit) → server applies to DocState → `applied` +
   datalogger verdict `committed`. `scratch` → verdict `scratched`.
3. Keyboard `edit` frames apply immediately (verdict `committed`,
   source `keyboard`); if they modify a line committed < N s ago, logger marks
   the earlier record `edited_after` with the corrected LaTeX as gold.
4. `undo` pops DocState history → `applied` with full new `doc_context`
   (client re-renders whole doc from `doc_context` on every `applied` — the
   server is the single source of truth).
