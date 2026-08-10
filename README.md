# Listening to the Network: 5G Root-Cause Diagnosis with an Edge-Sized LLM

Every second, telecom networks generate millions of log lines. Buried in that noise,
the network is describing exactly what is wrong — a mistilted antenna, a missing
neighbor-cell configuration, a congested control channel. The bottleneck is not data;
it is that no human can listen fast enough. A single field engineer may be responsible
for hundreds of sites, and every hour of manual log-digging is an hour of downtime for
communities that depend on the network for connectivity, mobile money, and emergency
services.

This project builds a diagnosis system around a single **4B-parameter open-source
LLM (Qwen3-4B)** — small enough to run on a constrained edge server — that reads raw
5G drive-test telemetry and answers, in plain language, *why* throughput collapsed on
a given road section.

**Independently benchmarked at 99.1% accuracy (855/863 held-out fault scenarios),
finishing 2nd of 130 teams in the Cassava AI Root Cause Detective challenge
(Deep Learning Indaba 2026) — one question behind first place, using 3 leaderboard
submissions.**

## The core finding: small models can't read tables, but they can read measurements

The naive approach — fine-tune the LLM on raw telemetry tables — fails almost
completely. A 4B model asked to reason over a 10-row table of RSRP/SINR/GPS readings
scored **at chance level (~12%)**: it cannot reliably do the arithmetic (haversine
distances, coverage geometry, mod-30 collisions) that diagnosis requires, so it
pattern-matches on surface text instead.

The fix is a division of labor that mirrors how human experts actually work:

1. **Deterministic code does the measuring** — parsing the telemetry and computing
   the quantities an engineer would reach for: UE-to-cell distance, downtilt coverage
   edge, neighbor signal gaps, handover counts, scheduling statistics.
2. **The verbalized measurements replace the raw tables in the prompt** — a compact
   "computed diagnostic summary" in plain language.
3. **The model does the diagnosing** — LoRA-fine-tuned on ~3k labeled faults, it
   learns to weigh the measurements and generate the root cause, with
   self-consistency voting over multiple seeded samples.

Same model, same data: **~12% → 99% accuracy.** Measurement belongs in code;
judgment belongs in the model. The model's held-out error pattern confirms it is
genuinely judging, not echoing thresholds: its residual mistakes concentrate on the
one boundary that is legitimately ambiguous (weak coverage at the far end vs. a
stronger overshooting neighbor).

## Architecture

```
test questions ─┬─ general-knowledge ────► Qwen3-4B (base, thinking mode) ──► seeded 12-vote majority ─┐
                │                                                                                      ├─► diagnosis
                └─ telemetry faults ─► parser ─► feature verbalizer ─► prompt                          │
                                       (measurements only)              │                              │
    labeled faults ─► SFT data (+synthetic scenarios) ─► LoRA fine-tune ─► Qwen3-4B ───────────────────┘
                                                          (same model)     5-vote majority
```

Design decisions worth noting:

- **One model, two modes.** General-knowledge questions are answered by the *base*
  weights in thinking mode — never fine-tuned — so domain specialization cannot
  erode general capability (catastrophic forgetting is avoided by construction, not
  by mitigation).
- **Unseen fault formats.** A second telemetry format (different KPIs, different
  fault taxonomy, zero training labels) is handled by synthesizing training
  scenarios programmatically from the format's structure — the model then
  generalizes to the real instances.
- **Robustness by voting.** Every answer is a majority over 5–12 seeded generations,
  which both improves accuracy and makes results reproducible across GPU hardware.
- **Answer-format generalization.** Training examples are augmented with shuffled,
  relabeled, and subsetted answer options, so the model binds its diagnosis to the
  *meaning* of an option rather than its position or label.

## Repository layout

```
5g_root_cause_diagnosis.ipynb   the solution, end to end, with full commentary
                                (start here)
solution/                       the same logic as importable modules (parsing,
                                feature computation, verbalization, SFT data)
sub/                            archival record: the two benchmarked runs as
                                executed (NVIDIA H200) — outputs, exact
                                submission files, per-question prediction logs,
                                trained LoRA adapters
Data/                           benchmark datasets (from the Zindi challenge
                                page; not redistributed)
README.pdf                      the reviewed solution documentation
```

## Reproducing

1. `pip install torch transformers peft numpy` (Python 3.12; GPU with ≥40 GB VRAM
   recommended — measured end-to-end ~5 h on an NVIDIA H200)
2. Obtain the datasets from the [challenge page](https://zindi.world/competitions/cassava-ai-root-cause-detective-hackathon)
   into `Data/`
3. Run `5g_root_cause_diagnosis.ipynb` top to bottom. All seeds are fixed; the
   trained adapter is on Hugging Face
   ([abdulsamod/qwen3-4b-5g-root-cause](https://huggingface.co/abdulsamod/qwen3-4b-5g-root-cause))
   for verification without retraining, and the executed originals in `sub/`
   show every expected output.

## Results

| Benchmark split | Accuracy |
|---|---|
| Public (259 scenarios) | 0.9961 (258/259) |
| Private (604 scenarios) | 0.9884 (597/604) |
| Combined | **99.1% (855/863)** |

Validated by independent code review (reproducibility, no rule-based answer
generation, no external data). Full journey — including the approaches that failed
and why — in the accompanying write-up. *(blog link goes here)*

## License

MIT
