# Pipette-cam hardware

How to put a real camera behind [`run.py`](run.py). The software reads plain
grayscale frames, so **any UVC (USB-video-class) camera works** - the same
`cv2.VideoCapture` other CPUs already speak. Nothing below is endorsed or tested
on hardware; treat every specific number as **VERIFY** against your own channel
spacing, z-travel, and well geometry before buying.

## What the camera has to see

Two related reads, at a short working distance (roughly **5-15 mm**, VERIFY for
your labware depth):

1. **Residual / dry, top-down** (`run.py`, `synth.tip_view`). A well bottom right
   after the supernatant / ethanol aspiration. The discriminating cue is a
   **specular glint** off any residual film, so lighting matters as much as the
   sensor: a **coaxial / ring LED** around the lens is what makes "wet vs dry"
   separable. Diffuse room light alone can wash the glint out - the `--glare`
   failure mode in the demo.
2. **Liquid height -> volume, side-on** (`run_qc.py`, `synth.column_view`). The
   **meniscus line** against the well wall, imaged from the side or via a small
   45 deg mirror below the deck. The reader turns the meniscus row into uL, so the
   same camera does ethanol-removal QC (is the column ~0?) and volume QC. Needs a
   clear side wall and even backlight; foam or a clinging droplet is where the
   classical row-threshold gets soft and a segmentation net earns its keep.

## Two mounting modes

1. **Channel-mounted borescope (per-tip, moving).** A thin borescope/endoscope
   rides on the pipetting channel and looks down next to the tip. Pro: images the
   exact well being serviced, travels with it. Con: the tip occupies the axis, so
   the cam sits at a small offset/angle - plan for that in the bracket.
2. **Fixed downward camera over a check station (static).** A board camera on a
   short gantry over one deck position; the channel presents each tip/well to it
   between steps. Pro: simplest mechanically, one calibration. Con: adds a move to
   a check station rather than imaging in place.

The top-down residual geometry (coaxial, well bottom in a fixed central disk)
matches mode 1. The side-on height read wants the well wall in frame - either
angle the mode-1 borescope or add a small mirror under a mode-2 station. For mode
2 the well won't be centered every time - add the `detect()` step from
[`well_detection`](../well_detection) to crop each well before the reader.

## The orthogonal checkpoint - plate-reader dye QC

A camera proves the *volume*; it cannot prove the *chemistry* (a well at the right
level with the wrong-concentration reagent looks identical). So `run_qc.py` pairs
the cam with a **sampled plate-reader dye QC**: transfer a small aliquot
(~2 uL, VERIFY) from a fraction of wells into a QC plate, add a readiness dye
(e.g. Rhodamine-B or an assay-appropriate reporter - chemistry is yours to
choose), and read absorbance/fluorescence on any plate reader. Signal tracks
concentration, so this catches the off-spec wells the camera is blind to. It is
**sampled**, not per-well, because it consumes sample and reader time - a hit
escalates to a fuller read and holds the batch. Any reader that exports per-well
values works; the demo models the signal, `dye_qc.py` is where a real reader's
output is parsed.

## Camera options (categories, not part numbers)

| Class | Search terms | Notes |
| --- | --- | --- |
| USB borescope / endoscope | "5.5mm / 8mm USB borescope, built-in LED, UVC" | Cheapest path; integrated ring LED; check focus at your working distance (many fix-focus at ~3-6 cm - VERIFY) |
| Industrial USB board camera | "USB2 UVC board camera module + M12 macro lens" | Better optics/exposure control; pick lens focal length for your WD; add your own ring light |
| Raspberry Pi camera + light | "Pi Camera + coaxial ring light" | Good for a fixed mode-2 station; needs a host, streams frames the same way |

All three enumerate as a normal webcam, so:

```python
import cv2
cam = cv2.VideoCapture(index)          # UVC device
ok, bgr = cam.read()
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
# gray -> verify.verify_well(gray, well, cal, pol)   # same call the demo uses
```

## Mounting

- **Bracket:** a 3D-printed clip to the channel carriage (mode 1) or a small gantry
  arm (mode 2). Keep the lens axis normal to the plate so the well bottom stays a
  circle, not an ellipse.
- **Fit:** confirm the cam body clears neighboring tips at your channel pitch
  (VERIFY - this is the usual blocker on 8-channel heads).
- **Focus & WD:** set focus at the well-bottom plane; a fixed-focus borescope only
  works if its native WD matches your labware - measure first.

## Calibration

One-point, exactly like `verify.calibrate`: image a **pipetted reference droplet
of known volume** once, read its wet fraction, and that fixes wet-fraction -> uL for
the run. Re-run it if you change labware, lens, or lighting.

## Where this plugs into PLR

The verdict crosses to PyLabRobot as a single `Verdict` / `ResidualLiquidError`;
the async call sites (grab a frame after `remove_supernatant`, re-aspirate on a
flag, halt on a gross residual) are written out as a documented seam in
[`plr_bridge.py`](plr_bridge.py). PLR's own volume tracker stays authoritative for
what was moved - the camera is the independent check that the move actually left
the well dry.
