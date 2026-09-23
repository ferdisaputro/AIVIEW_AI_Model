"""
Personality detection using the trained BiLSTM models in ../models.

Pipeline for a video file:
  1. Audio  : decode to PCM via PyAV -> VGGish embeddings -> (15, 128)
  2. Visual : sample 30 frames -> VGG-Face embeddings (DeepFace) -> (30, 4096)
  3. Score  : normalize with training-set stats, predict with both BiLSTM
              models, then late-fuse (average) into a single OCEAN result.

Usage:
    # Detect personality from a video
    python personality_detector.py --video path/to/video.mp4

    # (Re)compute + save normalization stats from the extracted training features
    python personality_detector.py --compute-stats
"""

import os
import sys
import glob
import json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

AUDIO_MODEL_PATH = os.path.join(ROOT, "models", "bilstm_audio_tf.keras")
VISUAL_MODEL_PATH = os.path.join(ROOT, "models", "bilstm_visual_tf.keras")
STATS_DIR = os.path.join(APP_DIR, "models")

TRAIN_FEATURES_ROOT = {
    "audio": os.path.join(PROJECT_ROOT, "output", "audio", "train"),
    "visual": os.path.join(PROJECT_ROOT, "output", "visual", "train"),
}

OCEAN = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]
SEQ_LEN = {"audio": 15, "visual": 30}
FEAT_DIM = {"audio": 128, "visual": 4096}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_audio_waveform(video_path):
    """Decode the first audio stream of a video to mono float32 PCM in [-1, 1]."""
    import av

    container = av.open(video_path)
    chunks, sample_rate = [], None
    try:
        for frame in container.decode(audio=0):
            if sample_rate is None:
                sample_rate = frame.sample_rate
            arr = frame.to_ndarray()
            if arr.dtype.kind in "iu":
                info = np.iinfo(arr.dtype)
                arr = arr.astype(np.float32) / max(1.0, float(info.max))
            else:
                arr = arr.astype(np.float32)
            chunks.append(arr)
    finally:
        container.close()

    if not chunks or sample_rate is None:
        raise ValueError("Video does not contain a decodable audio stream.")

    pcm = np.concatenate(chunks, axis=1 if chunks[0].ndim == 2 else 0)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=0)
    return pcm.astype(np.float32), int(sample_rate)


_VGGISH = None


def get_vggish():
    """Lazily build the VGGish embedding model (downloads weights on first call)."""
    global _VGGISH
    if _VGGISH is None:
        import torch
        import torchvggish

        with torch.no_grad():
            _VGGISH = torchvggish.vggish()
        _VGGISH.eval()
    return _VGGISH


def extract_audio_features(video_path):
    """Return (15, 128) audio embeddings from a video."""
    import torch
    import torchvggish

    pcm, sr = extract_audio_waveform(video_path)
    model = get_vggish()
    with torch.no_grad():
        examples = torchvggish.waveform_to_examples(pcm, sr, return_tensor=True)
        emb = model(examples).cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32).reshape(-1, FEAT_DIM["audio"])
    return _pad_trim(emb, SEQ_LEN["audio"], FEAT_DIM["audio"])


def vid_to_vec(video_path, output_dir=None, split="trainingData"):
    """Extract 30 face-aware VGG-Face embeddings from a video.

    The video is divided into 30 equal-length intervals. For each interval the
    frames are scanned sequentially and the first frame with a detected face is
    used. If no face is found across the whole interval, the last successfully
    detected face is reused as the interval's representation.

    If ``output_dir`` is set, each representation is also persisted as:
        - image  : ``<output_dir>/ImageData/<split>/<video_id>/frame{i}.jpg``
        - feature: ``<output_dir>/Features/visual/<split>/<video_id>/feature{i}.npy``

    Returns a ``(30, 4096)`` float32 array of embeddings.
    """
    import cv2
    from deepface import DeepFace

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read the video file.")

    interval = max(1, total_frames // SEQ_LEN["visual"])

    embeddings = []
    images = []
    last_vec = None
    last_frame = None

    for slot in range(SEQ_LEN["visual"]):
        slot_start = slot * interval
        slot_end = min(slot_start + interval, total_frames)

        found = False
        for frame_idx in range(slot_start, slot_end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.resize(frame, (480, 240))
            try:
                results = DeepFace.represent(
                    img_path=frame,
                    model_name="VGG-Face",
                    detector_backend="opencv",
                    enforce_detection=True,
                )
                vec = np.asarray(results[0]["embedding"], dtype=np.float32).reshape(-1)
                last_vec = vec
                last_frame = frame
                found = True
                break
            except Exception:
                continue

        if not found:
            if last_vec is None:
                continue
            vec = last_vec
            frame = last_frame
        else:
            vec = last_vec
            frame = last_frame

        embeddings.append(vec)
        images.append(frame)

    cap.release()

    if not embeddings:
        raise ValueError("No face could be detected in the video.")

    arr = np.stack(embeddings, axis=0)

    if output_dir:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        img_dir = os.path.join(output_dir, "ImageData", split, video_id)
        feat_dir = os.path.join(output_dir, "Features", "visual", split, video_id)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(feat_dir, exist_ok=True)
        for i, (img, feat) in enumerate(zip(images, embeddings)):
            cv2.imwrite(os.path.join(img_dir, f"frame{i}.jpg"), img)
            np.save(os.path.join(feat_dir, f"feature{i}.npy"), feat)

    return _pad_trim(arr, SEQ_LEN["visual"], FEAT_DIM["visual"])


def extract_visual_features(video_path):
    """Return (30, 4096) VGG-Face embeddings sampled from a video."""
    return vid_to_vec(video_path)


def _pad_trim(arr, seq_len, feat_dim):
    if len(arr) >= seq_len:
        return arr[:seq_len]
    pad = np.zeros((seq_len - len(arr), feat_dim), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------

def compute_norm_stats(mod):
    """Training-set per-dimension mean/std (identical to train_blstm.py).

    Two streaming passes (one file at a time) to stay within RAM; stacking all
    clips first would need ~3x the dataset size for visual (30, 4096).
    """
    root = TRAIN_FEATURES_ROOT[mod]
    paths = sorted(glob.glob(os.path.join(root, "*.npy")))
    if not paths:
        raise FileNotFoundError(f"Could not find training features in {root}")

    feat_dim = FEAT_DIM[mod]

    def good(arr):
        return arr.shape == (SEQ_LEN[mod], feat_dim) and np.isfinite(arr).all()

    count = 0
    s = np.zeros(feat_dim, dtype=np.float64)
    loaded = []
    for path in paths:
        try:
            arr = np.load(path)
        except (OSError, ValueError):
            continue
        if not good(arr):
            continue
        loaded.append(path)
        s += arr.sum(axis=(0, 1), dtype=np.float64)
        count += arr.shape[0] * arr.shape[1]
    if count == 0:
        raise FileNotFoundError(f"Could not find training features in {root}")

    mean = s / count
    ss = np.zeros(feat_dim, dtype=np.float64)
    for path in loaded:
        arr = np.load(path)
        d = arr.astype(np.float64) - mean
        ss += np.sum(d * d, axis=(0, 1))
    std = np.sqrt(ss / count) + 1e-8
    return mean, std


def save_norm_stats():
    os.makedirs(STATS_DIR, exist_ok=True)
    for mod in ("audio", "visual"):
        mean, std = compute_norm_stats(mod)
        out = os.path.join(STATS_DIR, f"norm_stats_{mod}.json")
        with open(out, "w") as f:
            json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
        print(f"saved {out} ({len(mean)} dims)")


def load_norm_stats(mod):
    p = os.path.join(STATS_DIR, f"norm_stats_{mod}.json")
    if not os.path.exists(p):
        save_norm_stats()
    with open(p) as f:
        d = json.load(f)
    return (np.asarray(d["mean"], dtype=np.float32),
            np.asarray(d["std"], dtype=np.float32))


# ---------------------------------------------------------------------------
# Model loading & prediction
# ---------------------------------------------------------------------------

def _prepare_tensorflow():
    """Set up TensorFlow/Keras for loading the trained models.

    The TF 2.21 pip wheel bundles its own CUDA libraries, so no manual
    preload is needed. (Preloading the venv's nvidia libs via ctypes as in
    bilstm_train_tf.ipynb interposes NVBLAS and can segfault this process.)
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(42)
    return tf


class PersonalityPredictor:
    def __init__(self):
        tf = _prepare_tensorflow()
        print("loading audio BiLSTM model ...")
        self.audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        print("loading visual BiLSTM model ...")
        self.visual_model = tf.keras.models.load_model(VISUAL_MODEL_PATH)
        self.stats = {mod: load_norm_stats(mod) for mod in ("audio", "visual")}

    def _predict(self, model, features, mod):
        mean, std = self.stats[mod]
        x = (features[None, ...].astype(np.float32) - mean) / std
        return model.predict(x, verbose=0)[0]

    def predict(self, video_path):
        print("extracting audio features ...")
        audio_feat = extract_audio_features(video_path)
        print("extracting visual features ...")
        visual_feat = extract_visual_features(video_path)

        audio = self._predict(self.audio_model, audio_feat, "audio")
        visual = self._predict(self.visual_model, visual_feat, "visual")
        fused = (audio + visual) / 2.0
        return {
            "audio": {t: float(s) for t, s in zip(OCEAN, audio)},
            "visual": {t: float(s) for t, s in zip(OCEAN, visual)},
            "fused": {t: float(s) for t, s in zip(OCEAN, fused)},
        }


_PREDICTOR = None


def get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = PersonalityPredictor()
    return _PREDICTOR


def warmup():
    """Pre-load Keras models and VGGish weights (first run downloads them)."""
    predictor = get_predictor()
    get_vggish()
    return predictor


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if "--compute-stats" in argv:
        save_norm_stats()
        return 0
    if "--video" in argv:
        video = argv[argv.index("--video") + 1]
    else:
        print(__doc__)
        return 2

    if not os.path.exists(video):
        print(f"Error: video not found: {video}")
        return 1

    predictor = warmup()
    result = predictor.predict(video)
    header = f"{'trait':<18}" + "".join(f"{m:>9}" for m in ("audio", "visual", "fused"))
    print("\n" + header)
    print("-" * len(header))
    for t in OCEAN:
        def pct(d):
            return f"{d[t] * 100:6.1f}%"
        print(f"{t:<18}" + "".join(f"{pct(d):>9}" for d in
                                   (result["audio"], result["visual"], result["fused"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())