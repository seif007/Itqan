"""Auto-resume driver for train.py on a flaky GTX 950M (intermittent CUDA
SIGSEGV exit 139). Loops train.py from its checkpoint until it terminates
cleanly (exit 0) or a safety limit of crashes is hit. The Phase 2 deliverable
in `dataset/` is never touched (Phase 3 always runs against `dataset_p13/`).

Usage:
    python src/auto_resume_train.py --dataset_dir ../dataset_p13 \
        --max_steps_per_rule 200 ... [any train.py arg]
"""
import os
import sys
import subprocess
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(HERE, "train.py")


def log(msg):
    print("[auto-resume] " + msg, flush=True)


def loadable_ckpt(dataset_dir):
    """Return (ok, epoch) or (False, None) if checkpoint is missing/corrupt."""
    ck = os.path.join(dataset_dir, "model_checkpoint.pt")
    if not os.path.exists(ck):
        return False, None
    try:
        import torch  # local import; train.py also needs it
        state = torch.load(ck, map_location="cpu")
        return True, state.get("epoch")
    except Exception as e:
        log("checkpoint load failed: %r" % e)
        return False, None


def main():
    ap = argparse.ArgumentParser(add_help=False)
    # pass any-args through to train.py + add our own --max_crashes
    ap.add_argument("--max_crashes", type=int, default=40,
                    help="give up after this many crash+resume attempts")
    known, train_args = ap.parse_known_args()

    log("auto-resume driver starting (max %d crashes before giving up)" %
        known.max_crashes)

    crashes = 0
    while True:
        cmd = [sys.executable, "-u", TRAIN_PY] + train_args
        log("launching train.py: %s" % (" ".join(cmd)))
        start = time.time()
        rc = subprocess.call(cmd)
        elapsed = int(time.time() - start)

        if rc == 0:
            log("train.py exited 0 (clean). training finished.")
            log("ph3 best.pt at %s" % os.path.join(train_args[
                train_args.index("--dataset_dir") + 1
                if "--dataset_dir" in train_args else 0], "model_best.pt"))
            return 0

        # non-zero exit code; SIGSEGV is 139 (unsigned) / -11 signed
        # subprocess.call returns the low byte, so SIGSEGV -> 11.
        log("train.py exited %d after %ds" % (rc, elapsed))

        ok, epoch = loadable_ckpt(train_args[
            train_args.index("--dataset_dir") + 1
            if "--dataset_dir" in train_args else ".."])
        if not ok:
            log("checkpoint gone or corrupt -- cannot resume. quitting.")
            return rc
        log("checkpoint OK at epoch %d; safe to resume" % epoch)

        crashes += 1
        if crashes >= known.max_crashes:
            log("hit max_crashes=%d; stopping auto-resume. Phase 3 best at %s"
                " remains deliverable; Phase 2 in phase2_model_backup/ is the "
                "fallback." % (known.max_crashes, train_args[
                    train_args.index("--dataset_dir") + 1
                    if "--dataset_dir" in train_args else "."]))
            return rc

        # cool-down to release any transient GPU state before relaunch
        time.sleep(5)
        log("resume attempt %d/%d" % (crashes + 1, known.max_crashes))


if __name__ == "__main__":
    sys.exit(main())
