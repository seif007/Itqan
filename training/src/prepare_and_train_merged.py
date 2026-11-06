#!/usr/bin/env python
"""
Workstream A (merged) orchestrator.

Waits for shards 8-12 of the Tadabur corpus to finish downloading, then:
  1. rebuilds the training dataset from the FULL read-only source
     (parent tadabur_part/ contains shard0..shard12 -- walk_sidecars recurses),
  2. trains the TajweedNet hybrid model (Conv on GPU, Bi-GRU on CPU) which
     covers ALL rules including MADD.

Why MADD is included here: the original "improve all rules except MADD" plan
concerned the *label-quality* work (per-clip denoising). The merged rebuild
re-derives every rule's labels from the larger corpus, and MADD is trained
alongside the rest -- if the extra data lifts MADD above its prior ~0.95 it is
a pure win and is NOT excluded from training. (MADD's per-clip denoising was
already done in an earlier workstream.)

Source is never modified, copied, or moved -- only read.
"""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TADABUR_PART = os.environ.get(
    "TADABUR_PART",
    os.path.join(os.path.expanduser("~"), "Downloads", "practice_recitation-main",
                 "practice_recitation-main", "tools", "calib", "tadabur_part"),
)
TARGET_DIR = os.path.join(HERE, "..", "dataset_merged")
SHARDS_DONE = os.path.join(TADABUR_PART, "_shards_done.txt")
REQUIRED_SHARDS = [8, 9, 10, 11, 12]

# Training hyper-parameters (same hybrid config that completed workstream C).
TRAIN_ARGS = [
    "--dataset_dir", os.path.abspath(TARGET_DIR),
    "--device", "auto",
    "--batch_size", "32",
    "--epochs", "80",
    "--dropout", "0.3",
    "--weight_decay", "1e-3",
    "--patience", "8",
    "--max_steps_per_rule", "60",
    "--fresh",
]


def log(*a):
    print("[merged-orch]", *a, flush=True)


def shards_done():
    if not os.path.exists(SHARDS_DONE):
        return set()
    with open(SHARDS_DONE, encoding="utf-8") as f:
        return {int(x.strip()) for x in f if x.strip().isdigit()}


def wait_for_shards(poll=60, timeout=6 * 3600):
    log("waiting for shards", REQUIRED_SHARDS, "in", SHARDS_DONE)
    waited = 0
    while True:
        done = shards_done()
        missing = [s for s in REQUIRED_SHARDS if s not in done]
        if not missing:
            log("all required shards present:", sorted(done))
            return True
        if waited >= timeout:
            log("TIMEOUT: still missing", missing, "-- aborting")
            return False
        log("missing", missing, "(have %d shards) -- sleeping %ds"
            % (len(done), poll))
        time.sleep(poll)
        waited += poll


def rebuild():
    log("=== rebuild dataset from merged source ===")
    cmd = [sys.executable, os.path.join(HERE, "prepare_dataset.py"),
           "--tadabur_part", TADABUR_PART,
           "--target_dir", os.path.abspath(TARGET_DIR)]
    log(" ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        log("prepare_dataset.py failed rc=%d" % rc)
        sys.exit(rc)
    log("=== rebuild done ===")


def train():
    log("=== train hybrid model (Conv GPU / GRU CPU) ===")
    cmd = [sys.executable, os.path.join(HERE, "train.py")] + TRAIN_ARGS
    log(" ".join(cmd))
    # Stream output to a log so we can inspect progress; run in foreground so
    # this orchestrator only returns once training is complete.
    out_path = os.path.join(os.path.abspath(TARGET_DIR), "train_merged.log")
    err_path = os.path.join(os.path.abspath(TARGET_DIR), "train_merged.err")
    with open(out_path, "w", buffering=1) as fo, \
         open(err_path, "w", buffering=1) as fe:
        rc = subprocess.run(cmd, stdout=fo, stderr=fe).returncode
    log("train.py exited rc=%d (see %s)" % (rc, out_path))
    if rc != 0:
        sys.exit(rc)
    log("=== training done ===")


def main():
    if not wait_for_shards():
        sys.exit(1)
    rebuild()
    train()
    log("MERGED WORKSTREAM A COMPLETE.")


if __name__ == "__main__":
    main()
