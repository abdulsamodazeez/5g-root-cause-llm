"""Gradio demo: paste a 5G fault scenario, get a diagnosis with its reasoning.

Run locally (GPU recommended):  python app.py
Or deploy as a Hugging Face Space (ZeroGPU works for a 4B model).
"""
import gradio as gr
from diagnose import Diagnostician

ADAPTER = "abdulsamod/qwen3-4b-5g-root-cause"  # HF Hub; or a local adapter dir, or None for base
d = Diagnostician(adapter=ADAPTER)


def run(scenario):
    if not scenario.strip():
        return "", "", ""
    r = d.diagnose(scenario)
    votes = ", ".join(f"{k}: {v}" for k, v in r["votes"].items())
    return r["measurements"], f"Root cause: {r['diagnosis']}   (votes - {votes})", r["explanation"]


demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(lines=18, label="Fault scenario (question + options + telemetry tables)"),
    outputs=[
        gr.Textbox(label="What the system measured", lines=10),
        gr.Textbox(label="Diagnosis"),
        gr.Textbox(label="Model's reasoning", lines=4),
    ],
    title="5G Root-Cause Diagnosis — Edge-Sized LLM",
    description=("Paste a drive-test fault scenario. Deterministic code computes the "
                 "measurements an RF engineer would; a fine-tuned Qwen3-4B weighs them "
                 "and states the root cause. Benchmarked at 99.1% accuracy."),
)

if __name__ == "__main__":
    demo.launch()
