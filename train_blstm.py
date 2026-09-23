#!/usr/bin/env python3
"""Train separate BiLSTM personality models for the audio and visual modalities.

One model per modality:
  audio  : (batch, 15, 128)  VGGish embeddings  -> models/bilstm_audio_tf.keras
  visual : (batch, 30, 4096) VGG-Face embeddings -> models/bilstm_visual_tf.keras

Architecture (per modality, proven config from old/bilstm_train_tf.ipynb):
  Input -> Bidirectional(LSTM(hidden)) -> Dropout -> Dense(64, relu)
        -> Dropout -> Dense(5, sigmoid)

Labels: First Impressions pickle annotations (drop `interview`, keep 5 OCEAN
traits). Features are z-scored per dimension with train-only statistics; the
same stats are written to app_prediction/models/norm_stats_{mod}.json so the
inference app (personality_detector.py) normalizes identically.

Evaluation reports MAE, RMSE and R2 (overall + per trait) on the validation
split into models/bilstm_{mod}_tf_history.json.

Usage:
    python train_blstm.py                        # train both modalities
    python train_blstm.py --modality audio
    python train_blstm.py --modality visual --epochs 5
    python train_blstm.py --limit 128 --epochs 2 # smoke run
    python train_blstm.py --eval-test            # also evaluate on test split
"""

import argparse
import gc
import json
import math
import os
import pickle
import random
import sys
import zipfile

import numpy as np

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)

DEFAULT_FEATURES_ROOT = os.path.join(PROJECT_ROOT, "output")
DEFAULT_ANNOTATIONS_DIR = os.path.join(PROJECT_ROOT, "first-impressions", "annotations")
DEFAULT_MODEL_DIR = os.path.join(APP_ROOT, "models")
DEFAULT_STATS_DIR = os.path.join(APP_ROOT, "app_prediction", "models")

OCEAN = ["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]
SEQ_LEN = {"audio": 15, "visual": 30}
FEAT_DIM = {"audio": 128, "visual": 4096}
MODALITIES = ("audio", "visual")

DEFAULT_EPOCHS = {"audio": 90, "visual": 55}
DEFAULT_HIDDEN = 64
DEFAULT_DROPOUT = 0.3
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_PATIENCE = 10
SEED = 42


# ---------------------------------------------------------------------------
# Labels & features
# ---------------------------------------------------------------------------

def load_labels(annotations_dir, split):
    """Return {video_id: float32 (5,)} for one split.

    train -> train-annotation/annotation_training.pkl (plain file)
    val   -> val-annotation-e.zip  (read in-memory, never unzipped)
    test  -> test-annotation-e.zip (read in-memory, never unzipped)
    """
    if split == "train":
        path = os.path.join(annotations_dir, "train-annotation", "annotation_training.pkl")
        with open(path, "rb") as f:
            ann = pickle.load(f, encoding="latin1")
    elif split == "val":
        path = os.path.join(annotations_dir, "val-annotation-e.zip")
        with _open_zip_pkl(path, "annotation_validation.pkl") as f:
            ann = pickle.load(f, encoding="latin1")
    elif split == "test":
        path = os.path.join(annotations_dir, "test-annotation-e.zip")
        with _open_zip_pkl(path, "annotation_test.pkl") as f:
            ann = pickle.load(f, encoding="latin1")
    else:
        raise ValueError(f"unknown split: {split}")
    mapping = {}
    for trait in OCEAN:
        for clip, score in ann[trait].items():
            stem = clip[:-4] if clip.endswith(".mp4") else clip
            mapping.setdefault(stem, {})[trait] = float(score)

    labels = {}
    for stem, m in mapping.items():
        if all(t in m for t in OCEAN):
            labels[stem] = np.array([m[t] for t in OCEAN], dtype=np.float32)
    return labels


def _open_zip_pkl(zip_path, member):
    """Open a (possibly encrypted) member pkl of an annotation zip.

    The ChaLearn annotation zips are password-protected; the password is
    expected in a ``password.txt`` next to the zip and is read from memory
    (nothing is extracted to disk).
    """
    pwd_path = os.path.join(os.path.dirname(zip_path), "password.txt")
    try:
        with open(pwd_path, "rb") as f:
            pwd = f.read().strip()
    except OSError:
        raise RuntimeError(
            f"{zip_path} is password-protected; put the password in {pwd_path}"
        )
    return _ZipMember(zipfile.ZipFile(zip_path), member, pwd)


class _ZipMember:
    """Context manager closing both the zip and the open member file."""

    def __init__(self, zf, member, pwd):
        self._zip = zf
        self._member = zf.open(member, pwd=pwd)

    def __enter__(self):
        return self._member

    def __exit__(self, *exc):
        self._member.close()
        self._zip.close()



def feature_clips(features_root, mod, split):
    d = os.path.join(features_root, mod, split)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"feature dir not found: {d}")
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(d)
        if f.endswith(".npy")
    )


def load_features(features_root, mod, split, clips, dtype):
    """Stack <root>/<mod>/<split>/<id>.npy into (n, seq, feat) `dtype`.

    Preallocates the output array and fills it file-by-file so peak memory is
    one output array + one file (no list->stack doubling).
    """
    seq_len, feat_dim = SEQ_LEN[mod], FEAT_DIM[mod]
    X = np.empty((len(clips), seq_len, feat_dim), dtype=dtype)
    kept = []
    i = 0
    for stem in clips:
        path = os.path.join(features_root, mod, split, f"{stem}.npy")
        try:
            arr = np.load(path)
        except (OSError, ValueError):
            continue
        if arr.shape != (seq_len, feat_dim) or not np.isfinite(arr).all():
            continue
        X[i] = arr
        kept.append(stem)
        i += 1
    if i == 0:
        raise RuntimeError(f"no usable {mod} features in {os.path.join(features_root, mod, split)}")
    return X[:i], kept


def mem_available_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def pick_dtype(args, n_clips_by_split, mod):
    if args.dtype != "auto":
        return np.dtype(args.dtype)
    total_elems = sum(n_clips_by_split.values()) * SEQ_LEN[mod] * FEAT_DIM[mod]
    need32 = total_elems * 4 + 1_500_000_000  # float32 data + TF runtime
    avail = mem_available_bytes()
    if avail is None or need32 <= avail * 1.3:
        return np.dtype(np.float32)
    return np.dtype(np.float16)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(tf, seq_len, feat_dim, hidden, dropout, lr, weight_decay):
    inp = tf.keras.Input(shape=(seq_len, feat_dim))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(hidden))(inp)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    out = tf.keras.layers.Dense(5, activation="sigmoid", name="ocean_output")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay),
        loss="mse",
        metrics=["mae"],
    )
    return model


def make_sequence_class(tf):
    class _Seq(tf.keras.utils.Sequence):
        def __init__(self, X, y, batch_size, shuffle, seed=SEED):
            super().__init__()
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.rng = np.random.default_rng(seed)
            self.indices = np.arange(len(X))
            if shuffle:
                self.rng.shuffle(self.indices)

        def __len__(self):
            return math.ceil(len(self.X) / self.batch_size)

        def __getitem__(self, i):
            idx = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
            xb = self.X[idx]
            if xb.dtype != np.float32:
                xb = xb.astype(np.float32)
            return xb, self.y[idx]

        def on_epoch_end(self):
            if self.shuffle:
                self.rng.shuffle(self.indices)

    return _Seq


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    err = yp - yt
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(ss_tot > 0, 1.0 - ss_res / np.maximum(ss_tot, 1e-12), 0.0)
    return {
        "overall": {
            "mae": float(mae.mean()),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(r2.mean()),
        },
        "per_trait": {
            t: {"mae": float(mae[i]), "rmse": float(rmse[i]), "r2": float(r2[i])}
            for i, t in enumerate(OCEAN)
        },
    }


def print_metrics(title, m):
    print(f"{title}:")
    print(f"  {'trait':<18}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for t in OCEAN:
        d = m["per_trait"][t]
        print(f"  {t:<18}{d['mae']:10.4f}{d['rmse']:10.4f}{d['r2']:10.4f}")
    o = m["overall"]
    print(f"  {'overall':<18}{o['mae']:10.4f}{o['rmse']:10.4f}{o['r2']:10.4f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_modality(mod, args, tf):
    seq_len, feat_dim = SEQ_LEN[mod], FEAT_DIM[mod]
    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS[mod]

    print(f"\n=== {mod} ===")
    labels = {"train": load_labels(args.annotations_dir, "train"),
              "val": load_labels(args.annotations_dir, "val")}
    splits_needed = ["train", "val"] + (["test"] if args.eval_test else [])
    for s in splits_needed:
        if s not in labels:
            labels[s] = load_labels(args.annotations_dir, s)

    clips = {}
    for s in splits_needed:
        c = [x for x in feature_clips(args.features_root, mod, s) if x in labels[s]]
        if args.limit:
            c = c[: args.limit]
        clips[s] = c
        print(f"  clips {s}: {len(c)}")

    dtype = pick_dtype(args, {s: len(clips[s]) for s in splits_needed}, mod)
    print(f"  storage dtype: {dtype}")

    data = {}
    for s in ("train", "val"):
        data[s] = load_features(args.features_root, mod, s, clips[s], dtype)
    Xtr, kept_tr = data["train"]
    Xva, kept_va = data["val"]
    ytr = np.stack([labels["train"][c] for c in kept_tr])
    yva = np.stack([labels["val"][c] for c in kept_va])

    mean = np.mean(Xtr, axis=(0, 1), dtype=np.float64)
    std = np.std(Xtr, axis=(0, 1), dtype=np.float64) + 1e-8
    Xtr -= mean
    Xtr /= std
    Xva -= mean
    Xva /= std

    os.makedirs(args.stats_dir, exist_ok=True)
    stats_path = os.path.join(args.stats_dir, f"norm_stats_{mod}.json")
    with open(stats_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
    print(f"  saved {stats_path}")

    model = build_model(tf, seq_len, feat_dim, args.hidden, args.dropout,
                        args.learning_rate, args.weight_decay)
    model.summary()

    Seq = make_sequence_class(tf)
    train_ds = Seq(Xtr, ytr, args.batch_size, shuffle=True)
    val_ds = Seq(Xva, yva, args.batch_size, shuffle=False)
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=args.patience,
        restore_best_weights=True, verbose=1,
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=[es], verbose=2,
    ).history

    y_pred = model.predict(val_ds, verbose=0)
    val_metrics = regression_metrics(yva, y_pred)
    best_epoch = int(np.argmin(history["val_mae"])) + 1
    best_val_mae = float(np.min(history["val_mae"]))
    print(f"  best epoch {best_epoch}/{epochs}  val MAE {best_val_mae:.4f}")
    print_metrics(f"  {mod} val metrics", val_metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"bilstm_{mod}_tf.keras")
    model.save(model_path)
    print(f"  saved {model_path}")

    meta = {
        "modality": mod,
        "traits": OCEAN,
        "config": {
            "seq_len": seq_len,
            "feat_dim": feat_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": epochs,
            "patience": args.patience,
            "seed": SEED,
            "storage_dtype": str(dtype),
        },
        "data": {
            "train_clips": len(kept_tr),
            "val_clips": len(kept_va),
            "features_root": args.features_root,
        },
        "norm_stats": stats_path,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "val_metrics": val_metrics,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    if args.eval_test:
        Xte, kept_te = load_features(args.features_root, mod, "test",
                                     clips["test"], dtype)
        yte = np.stack([labels["test"][c] for c in kept_te])
        Xte -= mean
        Xte /= std
        test_ds = Seq(Xte, yte, args.batch_size, shuffle=False)
        test_metrics = regression_metrics(yte, model.predict(test_ds, verbose=0))
        print_metrics(f"  {mod} test metrics", test_metrics)
        meta["test_metrics"] = test_metrics
        meta["data"]["test_clips"] = len(kept_te)

    history_path = os.path.join(args.output_dir, f"bilstm_{mod}_tf_history.json")
    with open(history_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved {history_path}")

    del Xtr, Xva, ytr, yva, train_ds, val_ds, model
    gc.collect()
    return meta


def prepare_tf():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    # mixed_float16 only where it helps (GPU); CPU training stays float32
    try:
        if gpus:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        else:
            tf.keras.mixed_precision.set_global_policy("float32")
    except Exception:
        tf.keras.mixed_precision.set_global_policy("float32")
    tf.keras.utils.set_random_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    print("TF GPUs:", gpus or "none (CPU training)")
    return tf


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train separate audio/visual BiLSTM models.")
    parser.add_argument("--modality", choices=("audio", "visual", "both"), default="both")
    parser.add_argument("--features-root", default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--annotations-dir", default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--stats-dir", default=DEFAULT_STATS_DIR)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override epochs for the selected modality")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--limit", type=int, default=None,
                        help="max clips per split (smoke runs)")
    parser.add_argument("--eval-test", action="store_true",
                        help="also evaluate the saved model on the test split")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16"), default="auto",
                        help="feature storage dtype (auto picks float16 if RAM is low)")
    args = parser.parse_args(argv)

    modalities = MODALITIES if args.modality == "both" else (args.modality,)
    tf = prepare_tf()

    results = []
    for mod in modalities:
        results.append(train_modality(mod, args, tf))
        tf.keras.backend.clear_session()
        gc.collect()

    print("\n=== summary ===")
    for r in results:
        o = r["val_metrics"]["overall"]
        print(f"{r['modality']:<8} clips {r['data']['train_clips']}/"
              f"{r['data']['val_clips']} | best epoch {r['best_epoch']} | "
              f"val MAE {o['mae']:.4f} RMSE {o['rmse']:.4f} R2 {o['r2']:.4f} | "
              f"models/bilstm_{r['modality']}_tf.keras")
    return 0


if __name__ == "__main__":
    sys.exit(main())
