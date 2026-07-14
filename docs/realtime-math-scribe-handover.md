# Real-Time Math Scribe & Assistant — Project Handover

**Goal:** A locally-run, voice-driven math scribe. The user dictates mathematics which is transcribed live into LaTeX, then edits the document conversationally ("change the denominator to x squared") with changes committed or undone by voice. Built on a fine-tuned Gemma 4 12B running entirely on local hardware.

**Status:** Planning complete. Next step is Stage 1 of the synthetic data pipeline.

---

## 1. Model Selection

**Target model: Gemma 4 12B** (Apache 2.0, open weights).

Rationale:
- Only Gemma 4 E2B, E4B, and 12B have native audio support (31B and 26B-A4B are text/vision only).
- The 12B uses a **unified decoder-only architecture**: raw 16 kHz audio is sliced into 40 ms frames (640 floats each) and projected linearly into the same embedding space as text. There is **no separate audio encoder** (no conformer layers, unlike E2B/E4B).
- Consequence: LoRA/QLoRA on the standard linear layers updates the *entire* speech-to-LaTeX path end to end. No frozen encoder bottleneck, no special handling of audio modules.
- Native function calling and system prompt support — the agentic editing behavior reinforces an existing capability rather than teaching one from scratch.

Fallback/prototyping option: E4B (fits free Colab T4, 10 GB VRAM training) for validating the dataset recipe cheaply before committing to 12B runs. Note E4B routes audio through 12 conformer encoder layers, so results won't transfer 1:1.

## 2. Hardware & Training Environment

**GPU: RTX 5070 Ti, 16 GB VRAM** (Blackwell — fast bf16/fp4 paths, supported by Unsloth).

Approximate VRAM budget for 12B QLoRA:
- ~7 GB NF4 base weights
- ~1 GB LoRA adapters + 8-bit optimizer state
- Remainder for activations — sequence length and audio clip duration are the two pressure valves.

**Framework: Unsloth** (supports Gemma 4 text/vision/audio/RL fine-tuning, ~1.5x faster, ~60% less VRAM vs FA2 setups).

Starting training config:

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Gemma-4-12B-it",
    max_seq_length=1536,        # audio tokens inflate context; drop to 1024 if OOM
    load_in_4bit=True,          # QLoRA (fine for dense models; NOT for the MoE 26B-A4B)
)

model = FastModel.get_peft_model(
    model,
    r=16, lora_alpha=16,        # start small; scale rank only after pipeline is stable
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_dropout=0,
)

# SFTConfig essentials
per_device_train_batch_size=1
gradient_accumulation_steps=16      # effective batch 16
gradient_checkpointing=True         # or Unsloth's own mode
optim="paged_adamw_8bit"
learning_rate=2e-4                  # cosine decay
bf16=True
num_train_epochs=1-2                # synthetic data is repetitive; overfits fast
```

Training rules of thumb:
- Mask the prompt (audio + instruction + doc context) from the loss — train only on the response.
- Keep training clips **under ~10 seconds** (matches VAD-chunked inference; a 5 s utterance ≈ 125 audio tokens at 40 ms/frame).
- A loss of 13–15 early on is a known, normal quirk of Gemma 4 multimodal training (also seen on Gemma 3n, Llama Vision). Don't panic.
- Train with a **system prompt that suppresses thinking mode** — fine-tuned Gemma 4 otherwise leaks reasoning traces into outputs (documented in the transcription fine-tune tutorial).
- Training occupies essentially all 16 GB; develop the dataset pipeline and inference app outside training sessions.
- Ballpark: ~10–20k clips at batch-1/accum-16 ≈ one night to a weekend per epoch on this card.

## 3. Runtime Architecture

Real-time behavior is an **inference-loop problem, not a training problem** — the 12B is turn-based, not streaming.

```
Mic ──► VAD segmentation (1–3 s utterances)
          │
          ├──► openWakeWord spotters (continuous, cheap):
          │      "commit" / "undo" / "scratch that" → handled by app instantly
          │
          └──► Gemma 4 12B (4-bit, ~8 GB inference):
                 input  = audio chunk + rendered doc state + system prompt
                 output = tool call (edit op) OR plain text (answer/clarification)
                              │
                              ▼
               App applies edit ops, owns all document state,
               transactions, commit/undo history
```

Key design decisions:
- **The model proposes; the app disposes.** `commit`/`undo` are *not* model tool calls — they're keyword-spotted intents handled by the app layer. This keeps them instant (even mid-generation) and avoids training the model on transaction semantics it would get wrong.
- Sub-second per-utterance latency for short outputs on this class of GPU "feels live" for dictation.
- Inference (≈8 GB) + KV cache + VAD + spotters all fit on the 5070 Ti simultaneously once training is done.

## 4. Tool Schema (deliberately minimal)

```json
append_math(latex)            // dictation continues
replace(target_id, latex)     // "change the denominator to x squared"
insert(after_id, latex)       // "add a step after line 3"
delete(target_id)
set_label(target_id, text)    // "call that equation 2"
```

- The app renders document context into the prompt with **stable IDs**:
  `[e1] \frac{1}{2}mv^2   [e2] E = mc^2` → model emits `replace("e2", ...)`.
- Use **Gemma 4's own function-calling chat template** for targets. Do not invent a custom JSON format — that discards the pretrained tool-calling prior.
- Ambiguous references ("change that exponent" with multiple candidates) → correct output is a plain-text clarifying question, not a guessed tool call. This must be in the training data.

## 5. Dataset Design

One unified dataset; every example has the same shape:
**(audio, document state, system prompt) → (tool call | text)**

Target mix:

| Share | Category | Notes |
|-------|----------|-------|
| ~50% | Pure dictation → `append_math` | Must include *varied* doc context (empty, partial, long) |
| ~35% | Edit commands → `replace`/`insert`/`delete`/`set_label` | Generated by programmatic mutation (see §6) |
| ~10% | Conversational / multi-step turns | Questions, clarifications, mixed utterances |
| ~5%  | Negatives | Ambiguous refs → clarify; out-of-scope; math questions deserving text answers |

Anti-shortcut rules (important):
- Dictation examples **must also carry document context**, or the model learns "context present = edit mode" and mangles the doc mid-dictation.
- Include **mixed utterances**: "the integral from zero to infinity… actually make that from one" in one breath — train the model to emit what the user *meant*.
- Generate all categories from the **same underlying LaTeX corpus** and rotate TTS voices across all splits, so the model can't detect the category from phrasing/acoustic artifacts instead of intent.
- Generate whole small documents (3–8 numbered expressions) and derive dictation, edit, and mixed examples from each — gives consistent doc-context formatting and stable IDs across all example types for free.

## 6. Synthetic Data Pipeline (Piper + openWakeWord + LLM)

Note: openWakeWord's own training pipeline is built on **piper-sample-generator**, so one TTS toolchain powers both the Gemma corpus and the wake-word spotters.

**Stage 1 — LLM generation (text only).**
LLM produces structured records:
`{latex, spoken_form(s), doc_context, mutation_instruction, ground_truth_tool_call}`
- **Validate all LaTeX mechanically** (KaTeX/latexmk compile; SymPy parse for expressions); discard failures — never train on hallucinated broken LaTeX.
- Request **multiple spoken forms** per expression ("one half" / "one over two"; "x squared" / "x to the power of two") — verbalization diversity drives linguistic robustness.
- Spoken forms must be fully TTS-ready: no symbols, no ambiguous digits, Greek letters spelled out. Piper reads exactly what it's given.
- Edit examples: take a generated LaTeX doc, programmatically mutate it, TTS the natural-language instruction describing the *reverse* mutation → perfect ground-truth tool calls for free.

**Stage 2 — Piper rendering + augmentation.**
- 16 kHz output (matches Gemma's audio frontend).
- Rotate 10+ Piper voices (mixed genders/dialects).
- Augment aggressively: speed perturbation (0.9–1.1x), pitch shift, background noise mixing, room impulse responses. Piper output is clean/synthetic — augmentation closes most of the real-mic gap.

**Stage 3 — Real-voice anchor.**
- Record the primary user reading 500–1000 corpus items (a prompt-flashing recorder script; a couple of evenings).
- Mix in at ~10–15% of training data.
- Hold a slice out as the **only trusted eval set** — synthetic-audio eval numbers will flatter the model; real-voice numbers won't.

**Stage 4 — openWakeWord spotters.**
- Train small spotters for "commit" / "undo" / "scratch that" via piper-sample-generator.
- Use the user's own math-dictation recordings as **negative** examples (dictation that must never trigger a commit).
- Run continuously on the mic stream at negligible cost alongside the 12B.

## 7. Evaluation

Three crisp metrics instead of fuzzy transcript matching:
1. **Tool-name accuracy** — did it pick the right operation (or correctly answer in text)?
2. **Target-ID accuracy** — did it edit the right element?
3. **Payload correctness** — exact-match LaTeX *plus* SymPy-equivalence (so `\frac{1}{2}` vs `1/2` both count).

Evaluate on the held-out **real-voice** set. Track per-category (dictation / edit / conversational / negative) to see exactly which behavior is failing.

## 8. Roadmap

1. **Stage 1 generator** — LLM prompts + LaTeX validation harness. *(Next task.)*
2. **Stage 2 renderer** — Piper batch rendering + augmentation script.
3. **Pilot fine-tune** — small corpus (~2k clips) on E4B or 12B; sanity-check loss behavior, thinking-mode suppression, tool-call formatting.
4. **Real-voice recording sprint** + full corpus generation (10–20k clips).
5. **Full 12B QLoRA run** (1–2 epochs) + evaluation on real-voice holdout.
6. **openWakeWord spotters** for commit/undo/scratch-that.
7. **Inference app** — VAD loop, doc-state renderer with stable IDs, edit-op executor, commit/undo transaction layer.
8. **Live iteration** — collect real usage failures, fold corrections back into the dataset, re-tune.

## 9. Known Risks & Gotchas

- **Thinking-mode leakage** into fine-tuned outputs — mitigate with suppressing system prompt at train *and* inference time.
- **High multimodal loss (13–15)** is normal; use the eval metrics, not train loss, to judge progress.
- **Synthetic-only eval is misleading** — always report the real-voice holdout numbers.
- **Category-detection shortcuts** — enforce the anti-shortcut dataset rules in §5.
- **QLoRA is not recommended for the MoE variant (26B-A4B)** — irrelevant for the 12B, but noted in case of future model changes.
- **VRAM contention** — training saturates the 16 GB card; schedule around it.
- Unsloth's 12B support is newer than E2B/E4B/26B/31B — check current Unsloth docs/notebooks before assuming identical loaders and examples.

## 10. Key References

- Unsloth Gemma 4 training docs: https://unsloth.ai/docs/models/gemma-4/train
- Gemma 4 transcription fine-tune walkthrough (thinking-mode issue, Gradio eval harness): https://debuggercafe.com/fine-tuning-gemma-4-for-transcription/
- Gemma 4 12B architecture deep-dive (unified audio/vision decoder): https://www.labellerr.com/blog/gemma-4-12b-run-locally-and-fine-tune/
- Google Gemma fine-tuning docs: https://ai.google.dev/gemma/docs/tune
- Apple Silicon multimodal tuner (alt path, no CUDA): https://github.com/mattmireles/gemma-tuner-multimodal
- Piper TTS: https://github.com/rhasspy/piper
- openWakeWord (+ piper-sample-generator): https://github.com/dscripka/openWakeWord
