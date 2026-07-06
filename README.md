# lab-cv

Computer vision for the bench - **classical baselines with the exact seam where
a learned detector, SAM2, or a VLM drops in.** Every demo is clean-room: it
generates its own synthetic data, plants known ground truth, runs the method
blind, and scores recovery. CPU-only, no GPU, no downloads, no real media.

The through-line is **video-verified execution**: find the instances, verify the
state of each, hold identity across frames, and escalate the ambiguous ones - the
CV side of turning tacit lab know-how into reproducible, checkable automation.

```
frames ─▶ detect()      where are the instances?      demos/well_detection
       ─▶ classify()    what state is each in?         demos/well_state
       ─▶ track()       same identity across frames?   demos/roi_tracking
       ─▶ label()       name / adjudicate the unsure    demos/vocab_vlm
                        every step scored vs the plant  eval/metrics.py
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all            # unit tests + all five demos, each scored vs its plant
```

Each demo also runs on its own (`python demos/well_state/run.py`), writes a QC
plot to `output/`, and prints a one-line pass/fail against ground truth.

## Demos - with the numbers they actually print

All figures below are printed by the scripts in this repo on synthetic data with
the classical baseline (no models installed). Re-run `make all` to reproduce.

| Demo | What it shows | Baseline result | Learned-model seam |
| --- | --- | --- | --- |
| **[well_detection](demos/well_detection)** | instance detection + COCO scoring | clean plate **AP@0.5 = 1.00**, AP@[.5:.95] = 0.70, precision 0.94, recall 1.00; with `--occluder` a "hand" hides wells -> **recall 0.96**, AP@0.5 0.95 | RF-DETR / RT-DETRv4 behind `detect(model="rfdetr")` |
| **[well_state](demos/well_state)** | per-instance state + confidence (spatial verification) | **48 instances, ~2-3 ms/frame, accuracy 1.00**, clean confusion matrix; `--partial 4` under-fills wells -> **4 low-confidence QC flags** | learned classifier / VLM, same `classify()` call |
| **[roi_tracking](demos/roi_tracking)** | identity across a deck video | localization **IoU 0.93**; crossing paths -> **2 ID switches** (greedy IoU has no memory), parallel -> **0** | SAM2 video memory behind `tracker.SAM2Tracker` |
| **[tacit_gap](demos/tacit_gap)** | where the camera passes but the chemistry fails | every well reads filled -> spatial verifier says GO, catching **0/6** wrong-concentration wells; a paired ground-truth readout catches **6/6** -> composed layer clears 42/48 | orthogonal validation readout (dye / plate-reader), same interface |
| **[morphokinetics](demos/morphokinetics)** | timing recovery under crowding | separable -> **7/7 events, MAE 0 min**; `--crowding` packs blastomeres -> baseline **1/7, MAE 60 min** | RF-DETR + SAM2 (built out in the two demos above) |
| **[vocab_vlm](demos/vocab_vlm)** | detector -> VLM layering | open-vocab labeling **accuracy 1.00**; VLM escalated on only the low-confidence boxes -> **~52% of calls saved** | Qwen3-VL (vocabulary) / Gemini 3 (reasoning), offline mock backend by default |

## How it's built

- **Plant-and-recover.** Nothing is asserted that isn't scored. Each demo draws
  the ground truth, hides it from the method, and reports error against it.
- **One interface per stage.** `detect()`, `classify()`, `tracker`, and the VLM
  `label_regions()` each expose a single call site, so the classical baseline and
  the learned model are swappable by a flag - the eval harness never changes.
- **Honest failure is the demo.** The occluder, the crossing, and `--crowding`
  are there to *break* the baseline. That break is precisely where a learned
  detector / SAM2 / VLM earns its keep - shown, not hidden.
- **Shared, unit-tested scoring.** IoU, precision/recall, AP@0.5, AP@[.5:.95],
  confusion matrix, and ID-switch counting live in one auditable file,
  [`eval/metrics.py`](eval/metrics.py), with hand-checked tests in
  [`eval/test_metrics.py`](eval/test_metrics.py) (`make test`).

The low-confidence flag in `well_state` is the point where CV meets lab
reproducibility: a step can *look* executed and still be wrong ("the motions look
right, the chemistry's off"). Spatial verification catches the visible half and
flags the ambiguous half for orthogonal QC, rather than passing it silently.
`tacit_gap` takes that to its conclusion: spatial AI proves the motion, a
ground-truth readout proves the chemistry, and only their composition is a real
trust layer - the camera cannot see a wrong concentration in a correctly filled
well, so something orthogonal has to.

## Real-model paths (optional)

Every result above is the classical baseline. The learned paths are guarded and
optional - install [`requirements-models.txt`](requirements-models.txt) only for
the seam you want (RF-DETR, SAM2, a VLM). Nothing here ships weights, and the
baselines never need them.

## Legacy ROI-motion pipeline

The original frame-to-frame absdiff pipeline is still here (`src/`, `make
synthetic`): synthetic deck video -> per-ROI brightness/motion -> QC plot validated
against a planted activity schedule. See [the pipeline steps](src).

## Note

Synthetic data throughout; results are for the classical baselines on that data,
generated on the fly from a fixed seed. No real or proprietary media is included
or required. The learned-model paths are documented seams behind the same
interfaces.

## License

[MIT](LICENSE)
