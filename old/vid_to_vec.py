#!/usr/bin/env python3
"""Extract per-video VGG-Face embeddings for the First Impressions dataset.

Replaces the notebook pipeline (random seeks + one feature{i}.npy per frame + per-video dirs).
For each video a single (30, 4096) float32 features.npy-equivalent file is produced directly as
<output_root>/<split>/<video_id>.npy.

Strategy per video
------------------
- 30 target frame indices selected evenly with np.linspace(0, total_frames-1, 30).
- The video is decoded once, sequentially (no seeks). Each target is resolved on its first
  occurrence: detection via DeepFace at the target => success; failure with a previous valid
  feature => fallback to the previous feature; failure before any valid feature => the target is
  marked pending and every further frame is probed until the first success, which backfills all
  pending targets (forward-fill).
- The VGG-Face model is built once per process and reused for every frame.

Output
------
- <output_root>/<split>/<video_id>.npy          (30, 4096) float32, only for VALID videos
- <output_root>/debug/<split>/<video_id>/frame{slot}.jpg 30 frames with face bbox, only for the
  first MAX_DEBUG_VIDEOS videos per run
- <output_root>/log.csv                        append-only process log / resume source

Resume
------
A video is skipped iff its latest log.csv row has status==VALID. INVALID rows are reprocessed
(production may improve); --force reprocesses everything.

Usage:
    python vid_to_vec.py --dataset-root /path/to/first-impressions --output-root /path/out
    python vid_to_vec.py --dataset-root ... --output-root ... --splits train --limit 5 --force
"""

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

NUM_SAMPLES = 30
MAX_DEBUG_VIDEOS = 10
MAX_FAILED_FRAMES = 8
FRAME_SIZE = (480, 240)
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv"

LOG_COLUMNS = [
    "timestamp",
    "process_time",
    "video_id",
    "split",
    "frames",
    "successes",
    "failures",
    "fallbacks",
    "forward_fills",
    "status",
    "reason",
]


def now_ts():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def make_dirs(output_root):
    os.makedirs(output_root, exist_ok=True)


def read_log(output_root):
    """Return {video_id: last row} from <output_root>/log.csv."""
    path = os.path.join(output_root, "log.csv")
    if not os.path.exists(path):
        return {}
    latest = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("video_id"):
                    latest[row["video_id"]] = row
    except (OSError, csv.Error):
        return latest
    return latest


def append_log(output_root, row):
    path = os.path.join(output_root, "log.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def get_video_files(dataset_root, split):
    split_dir = os.path.join(dataset_root, split)
    if not os.path.isdir(split_dir):
        print(f"Warning: split dir not found: {split_dir}")
        return []
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg")
    return sorted(
        p for p in glob.glob(os.path.join(split_dir, "*"))
        if p.lower().endswith(exts)
    )


def sample_indexes(total_frames, n):
    return np.linspace(0, total_frames - 1, n, dtype=np.int32)


def represent_face(model, frame):
    """Return (embedding float32 (4096,), bbox dict) or (None, None) on detection failure."""
    import cv2
    from deepface import DeepFace

    resized = cv2.resize(frame, FRAME_SIZE)
    try:
        results = DeepFace.represent(
            img_path=resized,
            model_name=MODEL_NAME,
            model=model,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except Exception:
        return None, None
    if not results:
        return None, None
    vec = np.asarray(results[0]["embedding"], dtype=np.float32).reshape(-1)
    if vec.shape != (4096,) or not np.isfinite(vec).all():
        return None, None
    return vec, results[0].get("facial_area")


def draw_bbox(frame, bbox):
    import cv2

    annotated = cv2.resize(frame, FRAME_SIZE).copy()
    if bbox:
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("w", 0)
        h = bbox.get("h", 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return annotated


def process_video(video_path, model):
    """Return (features (30,4096) float32 or None, debug_frames or None, stats dict).

    Single sequential pass, no seeks. Targets resolved at first occurrence; early
    targets with no prior valid feature become pending and are forward-filled on the
    first later success; failed targets after a valid feature fall back to it.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None, None, {"reason": "unreadable_video"}

    targets = sample_indexes(total_frames, NUM_SAMPLES)
    features = [None] * NUM_SAMPLES
    debug_frames = [None] * NUM_SAMPLES
    pending = []

    successes = 0
    failures = 0
    fallbacks = 0
    forward_fills = 0

    last_vec = None
    last_annotated = None
    next_idx = 0
    frame_idx = -1

    while next_idx < NUM_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        at_target = frame_idx == targets[next_idx]

        vec, bbox = represent_face(model, frame)
        if vec is not None:
            successes += 1
            last_vec = vec
            last_annotated = draw_bbox(frame, bbox)
            if at_target:
                features[next_idx] = vec
                debug_frames[next_idx] = last_annotated
                next_idx += 1
            if pending:
                backfill = last_vec
                frame_for_pending = last_annotated
                while pending:
                    p = pending.pop(0)
                    features[p] = backfill
                    debug_frames[p] = frame_for_pending
                    forward_fills += 1
        else:
            failures += 1
            if not at_target:
                continue
            if last_vec is not None:
                features[next_idx] = last_vec
                debug_frames[next_idx] = last_annotated
                fallbacks += 1
            else:
                pending.append(next_idx)
            next_idx += 1

    cap.release()

    stats = {
        "frames": total_frames,
        "successes": successes,
        "failures": failures,
        "fallbacks": fallbacks,
        "forward_fills": forward_fills,
    }

    if all(f is not None for f in features):
        arr = np.stack(features, axis=0)
        return arr, debug_frames, stats

    if last_vec is None:
        stats["reason"] = "no_face_ever_detected"
    elif failures > MAX_FAILED_FRAMES:
        stats["reason"] = "too_many_failures"
    else:
        stats["reason"] = "no_face_ever_detected"
    return None, None, stats


def save_debug(output_root, split, video_id, debug_frames):
    if not debug_frames:
        return
    dbg_dir = os.path.join(output_root, "debug", split, video_id)
    os.makedirs(dbg_dir, exist_ok=True)
    for slot, frame in enumerate(debug_frames):
        if frame is None:
            continue
        import cv2

        cv2.imwrite(os.path.join(dbg_dir, f"frame{slot}.jpg"), frame)


def main():
    parser = argparse.ArgumentParser(description="Extract per-video VGG-Face embeddings.")
    parser.add_argument("--dataset-root", required=True,
                        help="Directory containing train/ and val/ video splits.")
    parser.add_argument("--output-root", required=True,
                        help="Where to write <split>/<video_id>.npy, debug/..., log.csv.")
    parser.add_argument("--splits", default="train,val", help="Comma-separated splits (default train,val).")
    parser.add_argument("--limit", type=int, default=None, help="Max videos per split to process.")
    parser.add_argument("--force", action="store_true", help="Reprocess VALID videos too.")
    args = parser.parse_args()

    from deepface import DeepFace

    make_dirs(args.output_root)
    log = read_log(args.output_root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print("building VGG-Face model ...")
    model = DeepFace.build_model(MODEL_NAME)
    print("model ready")

    debug_used = 0
    for split in splits:
        if os.path.isdir(os.path.join(args.dataset_root, split)):
            out_split_dir = os.path.join(args.output_root, split)
            os.makedirs(out_split_dir, exist_ok=True)

        videos = get_video_files(args.dataset_root, split)
        if args.limit:
            videos = videos[: args.limit]
        print(f"[{split}] {len(videos)} videos")

        for i, video_path in enumerate(videos, 1):
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            prev = log.get(video_id)
            if prev and prev.get("status") == "VALID" and not args.force:
                print(f"  skip {video_id} (VALID)")
                continue

            is_debug = debug_used < MAX_DEBUG_VIDEOS
            t0 = time.time()
            arr, debug_frames, stats = process_video(video_path, model)
            process_time = round(time.time() - t0, 3)

            if arr is not None:
                np.save(os.path.join(args.output_root, split, f"{video_id}.npy"), arr.astype(np.float32))
                status = "VALID"
                reason = ""
                if is_debug:
                    save_debug(args.output_root, split, video_id, debug_frames)
                    debug_used += 1
                print(f"  ok   {video_id} | {process_time}s | {stats}")
            else:
                status = "INVALID"
                reason = stats.get("reason", "unknown")
                print(f"  FAIL {video_id} | {reason} | {process_time}s")

            row = {
                "timestamp": now_ts(),
                "process_time": process_time,
                "video_id": video_id,
                "split": split,
                "frames": stats.get("frames", ""),
                "successes": stats.get("successes", ""),
                "failures": stats.get("failures", ""),
                "fallbacks": stats.get("fallbacks", ""),
                "forward_fills": stats.get("forward_fills", ""),
                "status": status,
                "reason": reason,
            }
            append_log(args.output_root, row)
            log[video_id] = row

    print("done")


if __name__ == "__main__":
    sys.exit(main())