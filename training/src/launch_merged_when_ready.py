#!/usr/bin/env python
"""
Waits for shard 12 to finish (appear in _shards_done.txt after extraction),
then runs the merged dataset rebuild + hybrid training.

The download process is separate and may still be extracting shard 12 when this
starts; we only proceed once 12 is recorded done. Training uses the same hybrid
config (Conv on GPU, Bi-GRU on CPU) that completed workstream C. Output goes to
dataset_merged/ so the completed C model in dataset/ stays untouched.
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
SHARDS_DONE = os.path.join(TADABUR_PART, "_shards_done.txt")
TARGET_DIR = os.path.abspath(os.path.join(HERE, "..", "dataset_merged"))


def log(*a):
    print("[merged-launcher]", *a, flush=True)


def shards_done():
    if not os.path.exists(SHARDS_DONE):
        return set()
    with open(SHARDS_DONE, encoding="utf-8") as f:
        return {int(x.strip()) for x in f if x.strip().isdigit()}


def wait_for_shard12(poll=30, timeout=3 * 3600):
    log("waiting for shard 12 in", SHARDS_DONE)
    waited = 0
    while True:
        if 12 in shards_done():
            log("shard 12 done (all 0-12 present: %s)" % sorted(shards_done()))
            return True
        if waited >= timeout:
            log("TIMEOUT waiting for shard 12 -- aborting")
            sys.exit(1)
        log("shard 12 not yet done (have %s) -- sleep %ds"
            % (sorted(shards_done()), poll))
        time.sleep(poll)
        waited += poll


def rebuild():
    log("=== rebuild merged dataset ===")
    cmd = [sys.executable, os.path.join(HERE, "prepare_dataset.py"),
           "--tadabur_part", TADABUR_PART,
           "--target_dir", TARGET_DIR]
    log(" ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        log("prepare_dataset.py FAILED rc=%d" % rc)
        sys.exit(rc)
    log("=== rebuild done ===")


def train():
    log("=== train hybrid model (Conv GPU / GRU CPU) ===")
    cmd = [sys.executable, os.path.join(HERE, "train.py"),
           "--dataset_dir", TARGET_DIR,
           "--device", "auto",
           "--batch_size", "32",
           "--epochs", "80",
           "--dropout", "0.3",
           "--weight_decay", "1e-3",
           "--patience", "8",
           "--max_steps_per_rule", "60",
           "--fresh"]
    log(" ".join(cmd))
    out = os.path.join(TARGET_DIR, "train_merged.log")
    err = os.path.join(TARGET_DIR, "train_merged.err")
    with open(out, "w", buffering=1) as fo, open(err, "w", buffering=1) as fe:
        rc = subprocess.run(cmd, stdout=fo, stderr=fe).returncode
    log("train.py exited rc=%d (see %s)" % (rc, out))
    if rc != 0:
        sys.exit(rc)
    log("=== merged training done ===")


def main():
    wait_for_shard12()
    rebuild()
    train()
    log("MERGED WORKSTREAM A COMPLETE.")


if __name__ == "__main__":
    main()
