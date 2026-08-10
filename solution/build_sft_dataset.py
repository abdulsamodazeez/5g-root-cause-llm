"""Build the SFT dataset for LLM-only distillation into Qwen2.5-1.5B-Instruct.

Outputs sft_family1.jsonl (+ optionally family2 synthetic) with
{"prompt": ..., "completion": ...} records. Completions carry a short
feature-grounded rationale and end with \\boxed{<label>} in the question's
own option labeling. Option shuffle/relabel/subset augmentation mirrors the
formats observed in the test set. Fully seeded.
"""
import csv
import json
import random
import re
import sys

from featmatrix import full_features
from feature_text import compact_question

csv.field_size_limit(sys.maxsize)
rng = random.Random(0)

CAUSE_TEXT = {
    "C1": "The serving cell's downtilt angle is too large, causing weak coverage at the far end.",
    "C2": "The serving cell's coverage distance exceeds 1km, resulting in over-shooting.",
    "C3": "A neighboring cell provides higher throughput.",
    "C4": "Non-colocated co-frequency neighboring cells cause severe overlapping coverage.",
    "C5": "Frequent handovers degrade performance.",
    "C6": "Neighbor cell and serving cell have the same PCI mod 30, leading to interference.",
    "C7": "Test vehicle speed exceeds 40km/h, impacting user throughput.",
    "C8": "Average scheduled RBs are below 160, affecting throughput.",
}
CANON = list(CAUSE_TEXT)


def rationale(f, cause):
    if cause == "C7":
        return f"The low-throughput rows show GPS speeds up to {f['speed_bad_max']:.0f} km/h, exceeding the 40 km/h limit."
    if cause == "C8":
        return f"The average scheduled RB count in the low-throughput rows is {f['rb_bad_mean']:.0f}, below 160."
    if cause == "C2":
        return f"The UE-to-serving-cell distance reaches {f['dist_bad_max']:.0f} m, beyond 1 km, indicating over-shooting."
    if cause == "C5":
        return f"The serving PCI changes {f['n_handover']:.0f} times during the drive, indicating frequent handovers."
    if cause == "C6":
        return "A strong neighboring cell shares the same PCI mod 30 as the serving cell, causing downlink interference."
    if cause == "C4":
        return "Multiple rows show non-colocated co-frequency neighbors within 6 dB of the serving cell, i.e. severe overlapping coverage."
    if cause in ("C1", "C3"):
        facts = (f"Key indicators: far point {f['dist_bad_max']:.0f} m vs coverage edge {f['cov_edge']:.0f} m "
                 f"(ratio {f['dist_over_edge']:.2f}); total downtilt {f['tilt']:.0f} deg; far-end RSRP min {f['rsrp_bad_min']:.1f} dBm; "
                 f"top-neighbor gap mean {f['top1gap_mean']:.1f} dB (max {f['top1gap_max']:.1f} dB).")
        verdict = ("Weighing these, downtilt-limited coverage at the far end fits best."
                   if cause == "C1" else
                   "Weighing these, a stronger neighboring cell delivering higher throughput fits best.")
        return f"{facts} {verdict}"
    return ""


def reformat_options(question, answer, style, rng):
    """Rewrite the option block with shuffled order/labels; return new question + new answer label."""
    lines = question.split("\n")
    opt_idx = [i for i, l in enumerate(lines) if re.match(r"^C[1-8]:\s", l.strip())]
    if len(opt_idx) != 8:
        return None
    causes = CANON[:]
    rng.shuffle(causes)
    if style == "subset":
        k = rng.choice([5, 6, 7, 8])
        keep = causes[:k]
        if answer not in keep:
            keep[rng.randrange(k)] = answer
        causes = keep
    prefix = rng.choice(["", "", rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")])
    new_opts, new_answer = [], None
    for j, c in enumerate(causes, 1):
        label = f"{prefix}{j}"
        new_opts.append(f"{label}: {CAUSE_TEXT[c]}")
        if c == answer:
            new_answer = label
    n = len(causes)
    out = []
    for i, l in enumerate(lines):
        if i == opt_idx[0]:
            out.extend(new_opts)
        if i in opt_idx:
            continue
        l2 = re.sub(r"following 8 potential root causes", f"following {n} potential root causes", l)
        out.append(l2)
    return "\n".join(out), new_answer


def main(holdout_ids=None):
    """holdout_ids: validation IDs to EXCLUDE from SFT (kept for honest eval)."""
    holdout_ids = holdout_ids or set()
    recs = []
    for path, label_col in [("../Data/train.csv", "answer")]:
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            recs.append((r["question"], r[label_col]))
    vq = {r["ID"]: r["question"] for r in csv.DictReader(open("../Data/validation_questions.csv"))}
    vt = {}
    for r in csv.DictReader(open("../Data/validation_target.csv")):
        vt[r["ID"].rsplit("_", 1)[0]] = r["Target"]
    for i, q in vq.items():
        if i in holdout_ids:
            continue
        recs.append((q, vt[i]))

    out = []
    for q, ans in recs:
        f = full_features(q)
        why = rationale(f, ans)
        # canonical form
        out.append({
            "prompt": compact_question(q),
            "completion": f"{why} The most likely root cause is {ans}. Final answer: \\boxed{{{ans[1]}}}",
        })
        # four augmented variants; C1/C3 (the hard boundary) get two extra
        styles = ["shuffle", "subset", "shuffle", "subset"]
        if ans in ("C1", "C3"):
            styles += ["shuffle", "subset"]
        for style in styles:
            ref = reformat_options(q, ans, style, rng)
            if ref is None:
                continue
            q2, lab = ref
            out.append({
                "prompt": compact_question(q2),
                "completion": f"{why} The most likely root cause corresponds to option {lab}. Final answer: \\boxed{{{lab}}}",
            })

    rng.shuffle(out)
    with open("sft_family1.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote sft_family1.jsonl: {len(out)} records", file=sys.stderr)


if __name__ == "__main__":
    main()
