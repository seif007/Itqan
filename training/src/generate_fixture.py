#!/usr/bin/env python3
"""
generate_fixture.py - Produce a numeric-parity fixture for the Kotlin port.

For a handful of REAL clips (one per rule), compute the exact model inputs
(logmel 40x200, dur, ev[5]) using train.py's own functions, then run the
TORCHSCRIPT traced model (model_traced.pt) to get ground-truth logits. Write
everything to dataset/parity_fixture.json so the Android JVM test can assert the
Kotlin port matches PyTorch within tolerance.

Run: python src/generate_fixture.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import train as T  # reuse clip_to_tensor, RULES, model defs

DS = os.path.join(HERE, "..", "dataset")


def main():
    best_path = os.path.join(DS, "model_best.pt")
    if not os.path.exists(best_path):
        print("ERROR: model_best.pt not found; run train.py first.")
        sys.exit(1)
    # Load into a CPU model instance (avoids the CUDA-baked traced graph).
    model = T.TajweedNet(rules=T.RULES, dropout=0.0)
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model = model.cpu()
    model.eval()

    fixture = []
    per_rule = 2  # clips per rule
    for rule in T.RULES:
        path = os.path.join(DS, "manifest_%s.jsonl" % rule)
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                if o.get("split") == "test":
                    rows.append(o)
        if not rows:
            rows = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        import random
        random.Random(42).shuffle(rows)
        chosen = rows[:per_rule]
        for o in chosen:
            x, dur, ev = T.clip_to_tensor(o["wav_abs"], o["start_ms"], o["end_ms"])
            with torch.no_grad():
                logits = model(x, dur, ev)
            lg = {r: float(logits[r][0]) for r in T.RULES}
            # x: (1, n_mels, T) -> (n_mels, T)
            lm = x[0].numpy().tolist()
            fixture.append({
                "rule": rule,
                "wav_abs": o["wav_abs"],
                "start_ms": o["start_ms"],
                "end_ms": o["end_ms"],
                "logmel": lm,           # [40][T] (T may be <200 if clip short; Kotlin pads)
                "dur": dur[0].item(),
                "ev": ev[0].numpy().tolist(),
                "expected_logits": lg,
            })
            print("[fixture] %-13s logits MADD=%.3f GHUNNAH=%.3f" %
                  (rule, lg["MADD"], lg["GHUNNAH"]))

    out = os.path.join(DS, "parity_fixture.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fixture, f)
    print("[fixture] wrote", out, "with", len(fixture), "clips")


if __name__ == "__main__":
    main()
