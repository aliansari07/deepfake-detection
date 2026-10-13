# Data splits

Every dataset split used by training and evaluation is a committed JSON file.
`src/dataset.py` reads the file named by `CONFIG["SPLIT_FILE"]` (override with
the `DD_SPLIT_FILE` environment variable) and never shuffles the cache at
runtime.

| File | Purpose | Cache | Channels | Identity-disjoint |
|---|---|---|---|---|
| `legacy_t4-20.json` | Reproduces every FYP-1 result (T4-20, B6, website numbers). Default. | `cache/` (v1, float32, repeat-padded) | BGR | **No** (audit-fyp2 F1) |
| `ffpp_official_v1.json` | FYP-2 training, model selection (`val`) and the one-time final evaluation (`test`). | `cache_v2/` (v2, uint8, zero-padded + mask) | RGB | **Yes** |
| `ffpp_official/{train,val,test}.json` | Official FaceForensics++ source-target pairs (360 / 70 / 70). | — | — | — |

## Source of the official pairs

`ffpp_official/*.json` are copied unchanged from the FaceForensics++ repository:
<https://github.com/ondyari/FaceForensics/tree/master/dataset/splits>
(Rössler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019).

## How `ffpp_official_v1.json` is built

```powershell
python preextract.py --cache-version 2 --workers 18 --threads 1   # face cache v2 (~50 min CPU)
python tools/build_split.py                                       # writes the split
python tools/build_split.py --check                               # re-verify disjointness
```

- A pair `(a, b)` places `original/a`, `original/b` and, for each of the five methods
  (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter), both `a_b` and `b_a`
  in the same split. No identity or source clip appears in two splits.
- **DeepFakeDetection is excluded** (decision D2, option B): its matching real
  footage is not part of the dataset, so the class would add a scene shortcut.
- Videos with no detected face are excluded from every split. Videos with fewer
  than `min_faces_train` (5) faces are excluded from `train` only and are kept, but
  listed under `flagged_low_faces`, in `val` and `test`. Every exclusion is listed
  with its reason under `excluded`.
- `cache_manifest_sha256` pins the cache manifest the split was built from.

## Rules

- Model selection uses `val` only. `DeepfakeDataset(split="test")` raises unless
  `allow_test=True` is passed for the final evaluation (FIX_PLAN P4.2).
- Never edit a split file by hand; rebuild it with the tool and commit the result.
- A model trained on `ffpp_official_v1.json` expects **RGB** input. The legacy
  inference path (`src/preprocess.py`, website) produces BGR for T4-20; switching the
  demo to a v2 model needs the RGB path (FIX_PLAN P3.6 / P6.2).
