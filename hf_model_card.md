---
license: apache-2.0
base_model: Qwen/Qwen3-4B
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
  - telecom
  - root-cause-analysis
  - 5g
  - edge-ai
---

# Qwen3-4B: 5G Root-Cause Diagnosis (LoRA adapter)

A LoRA adapter that turns [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) into a
5G network fault diagnostician: given verbalized measurements computed from raw
drive-test telemetry (distances, antenna coverage geometry, neighbor signal gaps,
scheduling statistics), it identifies and explains the root cause of a throughput
collapse.

**Benchmark: 99.1% accuracy (855/863 held-out fault scenarios), 2nd place of 130
teams, Cassava AI Root Cause Detective challenge (Deep Learning Indaba 2026).**

The design principle: a 4B model cannot reliably compute over raw telemetry tables
(it scores near chance), but it diagnoses expertly from *computed measurements*
placed in the prompt. Measurement belongs in code; judgment belongs in the model.
Full pipeline, parser, and feature verbalizer:
**[github.com/abdulsamodazeez/5g-root-cause-llm](https://github.com/abdulsamodazeez/5g-root-cause-llm)**

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "abdulsamod/qwen3-4b-5g-root-cause")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
```

Prompts should contain the fault scenario's answer options plus a computed
diagnostic summary (see `feature_text.py` in the GitHub repo, which produces it
from raw telemetry). The model answers with a reasoning sentence and
`\boxed{<option>}`. Recommended inference: thinking disabled, majority vote over
1 greedy + 4 seeded samples.

## Training

- LoRA r=16, α=32, dropout 0.05, all attention + MLP projections
- ~15k examples: ~3k labeled fault scenarios with feature-grounded rationales,
  answer-format augmentation, and programmatically synthesized scenarios for a
  second fault taxonomy with zero labels
- 3 epochs, lr 1e-4 cosine, fp32 master weights with fp16 autocast, prompt tokens
  masked from the loss; all seeds fixed
- Trained on an NVIDIA H200 (~2 h)

General-knowledge capability of the base model is unaffected by design: the
adapter is applied only for telemetry diagnosis, and general questions are served
by the base weights.
