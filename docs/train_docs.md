# BLSTM Training Docs (`train_blstm.ipynb`)

Reference for the training notebook and its script twin `train_blstm.py`.
Documents data flow, model, artifacts, recorded results, and known defects.

---

## 1. Purpose

Train **two independent** BiLSTM regressors, one per modality, predicting the 5
Big-Five (OCEAN) traits in `[0, 1]`:

| Modality | Feature source | Input shape `(seq, feat)` | Model file |
|---|---|---|---|
| `audio` | VGGish clip embeddings | `(15, 128)` | `models/bilstm_audio_tf.keras` |
| `visual` | VGG-Face clip embeddings | `(30, 4096)` | `models/bilstm_visual_tf.keras` |

There is **no fusion layer**. Each model sees one modality only; combining audio +
visual happens downstream in `app_prediction/personality_detector.py`.

Target vector order is fixed everywhere by `OCEAN`:

```
0 extraversion
1 neuroticism
2 agreeableness
3 conscientiousness
4 openness
```

---

## 2. Repository layout assumed

Paths are derived from `os.getcwd()`, so the notebook **must** be run with the
working directory set to `app/` (repo layout `skripsi/app`):

| Constant | Value |
|---|---|
| `APP_ROOT` | `.../skripsi/app` |
| `PROJECT_ROOT` | `.../skripsi` |
| `DEFAULT_FEATURES_ROOT` | `.../skripsi/output` |
| `DEFAULT_ANNOTATIONS_DIR` | `.../skripsi/first-impressions/annotations` |
| `DEFAULT_MODEL_DIR` | `.../skripsi/app/models` |
| `DEFAULT_STATS_DIR` | `.../skripsi/app/app_prediction/models` |

### Inputs

Features (produced by upstream notebooks `vid_to_wav_vec.ipynb` / `vid_to_vec.ipynb`):

```
output/
  audio/{train,val,test}/<clip_id>.npy   # (15, 128) float32
  visual/{train,val,test}/<clip_id>.npy  # (30, 4096) float32
```

Annotations (ChaLearn first-impressions):

```
first-impressions/annotations/
  train-annotation/annotation_training.pkl   # plain pickle
  val-annotation-e.zip                        # encrypted, member annotation_validation.pkl
  test-annotation-e.zip                       # encrypted, member annotation_test.pkl
  password.txt                                # zip password, read into memory
```

---

## 3. Notebook cell order

Run top to bottom; later cells depend on names defined earlier.

| Cell | Id | Role |
|---|---|---|
| 0 | `cell-0` | Markdown: architecture + data summary |
| 1 | `cell-1` | Imports, path constants, hyperparameter defaults, `SEED = 42` |
| 2 | `98b52d55` | Sanity echo of `DEFAULT_MODEL_DIR` |
| 3 | `cell-2` | `load_labels` |
| 4 | `cell-3` | `feature_clips`, `load_features`, `mem_available_bytes`, `pick_dtype` |
| 5 | `cell-4` | `build_model`, `make_sequence_class` |
| 6 | `cell-5` | `regression_metrics`, `print_metrics` |
| 7 | `cell-6` | `train_modality` (the whole per-modality pipeline) |
| 8 | `cell-7` | `prepare_tf` (TF import, GPU/mixed precision, seeding) |
| 9 | `cell-8` | Markdown: config notes |
| 10 | `cell-9` | `config = types.SimpleNamespace(...)` — the only user-facing knob cell |
| 11 | `cell-10` | Driver: `prepare_tf()` → loop modalities → clear session → summary table |
| 12 | `cell-11` | Markdown: artifact list |

`train_blstm.py` exposes the same pipeline as CLI flags (`--modality`, `--epochs`,
`--limit`, `--dtype`, `--eval-test`, ...), useful for unattended runs.

---

## 4. Data pipeline

### 4.1 Labels — `load_labels(annotations_dir, split)`

1. Pickles the annotation container: train, val, test from a plain `.pkl`.
2. `ann[trait]` is `{clip_name: score}`. Flattens into
   `mapping[stem][trait] = float(score)`, stripping a `.mp4` suffix.
3. Keeps only clips that have **all five** traits; value is `float32 (5,)` in
   `OCEAN` order.
4. Unknown split name raises `ValueError(f"unknown split: {split}")`.

`_ZipMember` is a context manager that closes both the member stream and the
`ZipFile` object, so repeated calls do not leak descriptors.

### 4.2 Clip intersection — `train_modality`

For each needed split: `feature_clips(...)` lists `.npy` stems (sorted), then the
list is intersected with `labels[split]`. Optionally truncated by `config.limit`.
Only clips present in **both** features and annotations survive — this is why the
audio model trains on 5962 of the available clips.

### 4.3 Feature loading — `load_features(...)`

* Preallocates `X = np.empty((n, seq, feat), dtype)` and fills **file by file**.
  Peak memory = one output array + one file; no `list.append` → `np.stack`
  doubling.
* Per file, a clip is **silently skipped** when:
  * `np.load` raises `OSError` / `ValueError` (corrupt/truncated file), or
  * `arr.shape != (seq_len, feat_dim)`, or
  * `np.isfinite(arr).all()` is `False` (NaN / Inf).
* Returns `(X[:i], kept)`. `kept` is the authoritative clip order, and the label
  tensors are stacked from `kept`, not from the requested clip list — this keeps
  features and labels aligned after skipping.
* `RuntimeError` if nothing usable is left.

### 4.4 Memory-aware dtype — `pick_dtype(...)`

Reads `MemAvailable` from `/proc/meminfo`. With `dtype="auto"`:

```
need32 = total_clips * seq * feat * 4  + 1.5e9   # float32 array + TF runtime reserve
float32 chosen  if  need32 <= 1.3 * MemAvailable
float16 chosen  otherwise
```

The `Sequence` casts each batch to `float32` before it reaches Keras, so a
float16-storage run still trains in float32.

### 4.5 Normalization

`mean` / `std` computed **over the train split only**, `axis=(0, 1)` in float64,
i.e. one scalar mean and std per feature dimension (`feat_dim` values), with
`std += 1e-8` to avoid divide-by-zero. Applied in place to train and val (and
test when enabled), then persisted to
`app_prediction/models/norm_stats_{mod}.json`.

Train-only stats prevent leakage; exporting them is what allows
`personality_detector.py` to normalize live features identically. That module can
also recompute the same stats itself (`compute_norm_stats`) if the JSON is
missing.

### 4.6 Batching

`make_sequence_class(tf)` builds a `keras.utils.Sequence`:

* owns a permutation of indices, reshuffled in `on_epoch_end` when `shuffle=True`
  (train only; val/test stay ordered);
* `__len__ = ceil(n / batch_size)`;
* `__getitem__` slices indices, casts the batch to `float32` if needed.

Streaming from the in-memory array avoids Keras copying the whole tensor set.

---

## 5. Model

```
Input(seq, feat)
  -> Bidirectional(LSTM(hidden=64))     -> (None, 128)      # concat fwd + bwd
  -> Dropout(0.3)
  -> Dense(64, relu)
  -> Dropout(0.3)
  -> Dense(5, sigmoid, name="ocean_output")                # (None, 5) in [0,1]
```

Compile: `Adam(learning_rate=1e-4, weight_decay=1e-5)`, `loss="mse"`,
`metrics=["mae"]`.

Parameter counts (`4*h*(feat+h+1)` per LSTM direction, doubled for bidirectional):

| Modality | LSTM | Dense(64) | Head | Total |
|---|---|---|---|---|
| audio | 98,816 | 8,256 | 325 | **107,397** (~420 KB) |
| visual | 2,130,432 | 8,256 | 325 | **2,139,013** |

The visual model is dominated by the 4096→64 input projection, i.e. capacity goes
into the embedding width, not the recurrence.

### Callback

`EarlyStopping(monitor="val_mae", patience=10, restore_best_weights=True)`.
Training halts after 10 epochs without a new best `val_mae`, and weights are
rolled back to the best epoch before saving.

### TF setup — `prepare_tf()`

* `TF_CPP_MIN_LOG_LEVEL=2`.
* `set_memory_growth(True)` on every visible GPU (wrapped in `try/except`).
* Mixed precision policy: `mixed_float16` when a GPU exists, else `float32`.
* `tf.keras.utils.set_random_seed(SEED)` plus `random.seed` / `np.random.seed`.
* Returned `tf` handle is threaded through the helpers, so TensorFlow is imported
  **once**, late, after configuration.

---

## 6. Evaluation

`regression_metrics(y_true, y_pred)` casts to float64 and returns
`{overall, per_trait}` with MAE, RMSE, R²:

* `per_trait.*` computed column-wise (per trait).
* `overall.mae` = mean of the 5 per-trait MAEs.
* `overall.rmse` = `sqrt(mean(err**2))` over **all** elements (pooled, not the
  mean of per-trait RMSE).
* `overall.r2` = **mean of the 5 per-trait R²**, not a pooled R². Read it as
  "average trait-level explanatory power"; it is slightly lower than a pooled R².
* R² degrades to `0.0` when a trait is constant in `y_true` (`ss_tot == 0`).

Headline numbers printed in the summary table come from `meta["val_metrics"]`,
i.e. from `model.predict` **after** weight restoration.

---

## 7. Artifacts

Written per modality:

| Path | Content |
|---|---|
| `models/bilstm_{mod}.tf.keras` | SavedModel-format Keras model |
| `models/bilstm_{mod}_tf_history.json` | `modality`, `traits`, `config`, `data`, `norm_stats` path, `best_epoch`, `best_val_mae`, `val_metrics`, `test_metrics` (optional), full `history` |
| `app_prediction/models/norm_stats_{mod}.json` | `{"mean": [...], "std": [...]}` per feature dim |

Downstream consumers (`app_prediction/personality_detector.py`):

```
models/bilstm_audio_tf.keras
models/bilstm_visual_tf.keras
app_prediction/models/norm_stats_{audio,visual}.json
```

Train/val arrays and the Keras session are released between modalities
(`del`, `clear_session()`, `gc.collect()`) because the visual float32 set is
large (~3.9 GB for train+val).

---

## 8. Configuration (cell-9)

```python
config = types.SimpleNamespace(
    modality="both",      # "audio" | "visual" | "both"
    epochs=None,          # None -> audio 90, visual 55
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-5,
    hidden=64,
    dropout=0.3,
    patience=10,
    limit=None,           # e.g. 128 -> smoke run
    eval_test=False,      # True -> also score the test split
    dtype="auto",         # "auto" | "float32" | "float16"
)
```

Useful recipes:

* Smoke test: `limit=128, epochs=2, modality="audio"`.
* Fast run: `epochs=20`.
* Check test metrics: `eval_test=True`.

Sizes for `limit=None` (float32): audio train 5962×15×128 ≈ 46 MB, val ≈ 15 MB;
visual train 5962×30×4096 ≈ 2.93 GB, val ≈ 0.98 GB. On a low-RAM box set
`dtype="float16"` (or let `"auto"` pick it) for the visual modality.

---

## 9. Recorded results

Run: `SEED=42`, float32 storage, 1 GPU present, `mixed_float16` policy.

### audio — 5962 train / 1987 val clips, 82 epochs executed, best epoch 72/90

| trait | MAE | RMSE | R² |
|---|---|---|---|
| extraversion | 0.0970 | 0.1222 | 0.3114 |
| neuroticism | 0.0960 | 0.1204 | 0.3451 |
| agreeableness | 0.0902 | 0.1135 | 0.1889 |
| conscientiousness | 0.0978 | 0.1234 | 0.3659 |
| openness | 0.0945 | 0.1194 | 0.2969 |
| **overall** | **0.0951** | **0.1198** | **0.3016** |

### visual — 5962 train / 1987 val clips, 55 epochs executed, best epoch 52

| trait | MAE | RMSE | R² |
|---|---|---|---|
| extraversion | 0.0900 | 0.1124 | 0.4176 |
| neuroticism | 0.0938 | 0.1182 | 0.3682 |
| agreeableness | 0.0870 | 0.1096 | 0.2445 |
| conscientiousness | 0.0875 | 0.1113 | 0.4843 |
| openness | 0.0903 | 0.1144 | 0.3544 |
| **overall** | **0.0897** | **0.1132** | **0.3738** |

Reading of the numbers:

* Visual beats audio on every aggregate (R² 0.374 vs 0.302, MAE 0.0897 vs
  0.0951) — face embeddings carry more personality signal than audio embeddings
  here.
* `agreeableness` is the hardest trait in both modalities (R² 0.19 / 0.24).
* Train MAE keeps improving (0.073 for audio at epoch 82, 0.043 for visual at
  epoch 55) while val MAE flattens around epoch ~30 — mild overfit. The 90/55
  epoch budgets plus patience 10 are
  generous; the best checkpoint is what gets saved, so it is not harmful, only
  slow.
* R² ~0.3-0.4 means the models explain roughly a third of trait variance. This is
  typical for first-impressions regression with pretrained embeddings, but it
  caps any downstream "personality score" claim.

---

## 10. Known defects / caveats

1. **Notebook cell-6 is broken as written** — `splits_needed = ["trainPresignedGetObject", "val", ...]`
   (line 319 of the notebook). `load_labels` accepts only `train`/`val`/`test`,
   and `output/<mod>/trainPresignedGetObject` does not exist (real dirs are
   `train`, `val`, `test`). The cell raises `ValueError: unknown split` before
   training starts. `train_blstm.py` has the correct `["train", "val"]`.
   Fix: replace that string with `"train"`, or edit the notebook from
   `train_blstm.py`, which is otherwise identical.
2. **`meta["best_val_mae"]` is the history minimum, not a re-measured value.** It
   is read from `history["val_mae"]` before `predict`. It happens to agree with
   the printed per-trait table to ~5 decimals because weights were restored, but
   the two are conceptually different quantities.
3. **`overall.r2` is a mean of per-trait R²**, which is not the pooled R² most
   readers assume. Document it in any thesis table that quotes it.
4. **float16 storage reduces normalization precision.** With `dtype="float16"`,
   `X -= mean` / `X /= std` run in float16, so normalized values carry ~1e-3
   relative error. Acceptable for training, but results will not match the
   float32 run exactly.
5. **Silent clip dropping.** Bad-shape / non-finite / unreadable `.npy` files are
   skipped without a warning; only the surviving count is printed. If the count
   drops unexpectedly, feature extraction is suspect.
6. **`std` epsilon is `1e-8`**, guarding division but not equal to skipping dead
   dimensions. A constant feature gets mean-subtracted to ~0 and divided by a
   near-zero std, which can amplify noise. Rare in practice with VGG embeddings.
7. **Path is CWD-derived.** Running the notebook from another directory silently
   points `PROJECT_ROOT` elsewhere and raises `FileNotFoundError` from
   `feature_clips`. Use `train_blstm.py` for anything non-interactive.
8. **`mixed_float16` + oneDNN warning.** On CPU-only hosts TF logs
   `oneDNN supports DT_HALF only on platforms with AVX-512`; harmless, and the
   policy falls back to float32 when no GPU is listed.

---

## 11. How to run

Notebook (must be in `app/`):

```bash
cd /path/to/skripsi/app
jupyter notebook train_blstm.ipynb
# edit config in cell-9, then run cells 1..10 in order
```

Script:

```bash
python train_blstm.py                      # both modalities, default schedules
python train_blstm.py --modality audio
python train_blstm.py --limit 128 --epochs 2     # smoke run
python train_blstm.py --eval-test --dtype float32
```

Encrypted zip splits need `first-impressions/annotations/password.txt`; otherwise
`_open_zip_pkl` raises `RuntimeError` naming the expected path.

---

## 12. Reproducibility notes

* `SEED = 42` is applied to `tf.keras.utils.set_random_seed`, `random`, and
  `np.random.seed`; the `Sequence` shuffles with its own
  `np.random.default_rng(42)`. Reruns on the same machine/TF build reproduce the
  reported numbers to ~1e-6.
* cuDNN/cuBLAS non-determinism and `mixed_float16` reduction order mean exact
  bitwise parity across machines is not guaranteed; expect ~1e-3 metric drift.
* Recorded per-epoch `loss`, `mae`, `val_loss`, `val_mae` are stored in
  `models/bilstm_{mod}_tf_history.json` under `history`.