# lab-cv

Offline computer-vision pipeline for ROI motion analysis of lab deck / protocol video. Runs entirely CPU-only with synthetic data -- no real video or downloads required.

## Quickstart

```bash
pip install opencv-python numpy pandas matplotlib
make synthetic
```

This generates a synthetic deck-camera video, extracts frames, builds a contact sheet, measures per-ROI brightness and motion, writes a dashboard summary, and produces a QC plot with validation.

## Pipeline

1. **Generate synthetic video** -- `src/generate_synthetic_video.py` writes a short MP4 with a static background and four ROI regions that brighten on a known schedule (simulating tip pickup, plate motion, etc.).

2. **Extract frames** -- `src/extract_frames.py` samples frames at `--every-sec` intervals.

3. **Contact sheet** -- `src/make_contact_sheet.py` tiles sampled frames into a single review image for quick visual QC.

4. **ROI motion analysis** -- `src/analyze_roi_motion.py` reads an ROI config (`example_roi_config.json`) and computes per-frame brightness and frame-to-frame absolute difference for each region, writing results to CSV.

5. **Dashboard summary** -- `src/dashboard_summary.py` copies the latest frame and writes a JSON summary of detected activity per ROI.

6. **QC plots** -- `src/plots.py` generates per-ROI brightness and motion traces, then validates detected motion peaks against the planted activity schedule.

## How it works

The core detection is frame-to-frame absolute difference within each ROI:

```python
# For each ROI crop (greyscale), compare to the previous frame
diff = np.abs(current_crop - previous_crop)
absdiff_mean = diff.mean()
absdiff_p95  = np.percentile(diff, 95)
# First frame has absdiff = 0 (no previous frame)
```

A region is considered "active" when `absdiff_mean` exceeds a threshold (default 5.0). The synthetic pipeline validates this by comparing detected activity against the known schedule and printing a one-line result.

## Run on your own video

```bash
python3 src/extract_frames.py protocol_video.mp4 --every-sec 2.0 --outdir frames
python3 src/make_contact_sheet.py --framedir frames --out output/contact_sheet.png
# Edit example_roi_config.json with x/y/w/h matching your camera view
python3 src/analyze_roi_motion.py --framedir frames --roi-config example_roi_config.json
python3 src/dashboard_summary.py
python3 src/plots.py --schedule ""
```

> Raw videos and extracted frames should never be committed to the repository.

## License

[MIT](LICENSE)
