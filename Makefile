PYTHON ?= python3

.PHONY: all test detection state tracking tacit pipette pipetteqc morphokinetics vocab synthetic clean

# --- CV portfolio: unit tests + every demo, each scored vs its plant ----------
all: test detection state tracking tacit pipette pipetteqc morphokinetics vocab

test:
	$(PYTHON) -m eval.test_metrics

detection:
	$(PYTHON) demos/well_detection/run.py

state:
	$(PYTHON) demos/well_state/run.py

tracking:
	$(PYTHON) demos/roi_tracking/run.py

tacit:
	$(PYTHON) demos/tacit_gap/run.py

pipette:
	$(PYTHON) demos/pipette_cam/run.py

pipetteqc:
	$(PYTHON) demos/pipette_cam/run_qc.py

morphokinetics:
	$(PYTHON) demos/morphokinetics/run.py

vocab:
	$(PYTHON) demos/vocab_vlm/run.py

# --- legacy ROI-motion pipeline (absdiff over a synthetic deck video) ---------
synthetic:
	$(PYTHON) src/generate_synthetic_video.py
	$(PYTHON) src/extract_frames.py videos/synthetic_deck.mp4 --every-sec 0.5
	$(PYTHON) src/make_contact_sheet.py
	$(PYTHON) src/analyze_roi_motion.py
	$(PYTHON) src/dashboard_summary.py
	$(PYTHON) src/plots.py

clean:
	rm -rf frames/ videos/ output/ **/__pycache__/ __pycache__/
