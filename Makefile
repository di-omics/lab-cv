PYTHON ?= python3

.PHONY: synthetic clean

synthetic:
	$(PYTHON) src/generate_synthetic_video.py
	$(PYTHON) src/extract_frames.py videos/synthetic_deck.mp4 --every-sec 0.5
	$(PYTHON) src/make_contact_sheet.py
	$(PYTHON) src/analyze_roi_motion.py
	$(PYTHON) src/dashboard_summary.py
	$(PYTHON) src/plots.py

clean:
	rm -rf frames/ videos/ output/
