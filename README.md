# sports (MVP)

This small package demonstrates pose detection using MediaPipe, a minimal IoU-based tracker, and optional Google Sheets upload.

Files of interest:

- `src/main.py` - Minimal runner (smoke test).
- `src/pose_detector.py` - MediaPipe Pose wrapper.
- `src/tracker.py` - Simple IoU tracker and iou utility.
- `src/sheets_uploader.py` - Google Sheets uploader using a service account.

## Quick start

1. Create and activate the venv (PowerShell):

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Launch the web trainer (requires a camera):

```powershell
python webapp\app.py --host 127.0.0.1 --port 5000
```

3. Open <http://127.0.0.1:5000> in Chrome/Edge. Allow camera access, choose a sport, and follow the animated demo. The UI streams your webcam to the server, overlays pose landmarks, and displays:

- Live confidence, FPS, and detected keypoint counts.
- Sport-specific rep counter and phase/status indicator.
- Running totals per sport for the current session.
- Ctrl+Q (or ⌘+Q on macOS) to gracefully shut down the server.

4. Recording & review:
	- Click **Start Recording** to create a JSONL session file under `webapp/sessions/` (metadata + per-frame counts/confidence).
	- Use **Stop Recording** to finalize; optional Google Sheets upload runs if `GOOGLE_APPLICATION_CREDENTIALS` and `SHEETS_ID` are set.
	- Visit <http://127.0.0.1:5000/viewer> to plot session keypoints vs. rep counts and view a quick summary of totals.

5. Run automated tests:

```powershell
pytest
```

6. Convert a recorded video (local file or HTTP(S) link) into an annotated GIF:

```powershell
python -m src.video_to_gif path\to\recording.mp4 -o output.gif --stride 2 --resize-width 720
```

To work from a YouTube link instead of a local file, pass the URL directly. The tool will fetch the clip via `yt-dlp`, process it frame by frame, and discard the temporary download when finished:

```powershell
python -m src.video_to_gif https://www.youtube.com/watch?v=dQw4w9WgXcQ -o output.gif --stride 3
```

> **Note:** `yt-dlp` is bundled in `requirements.txt`. Some formats may require `ffmpeg` on your `PATH` for muxing; install it if downloads fail.

The utility samples frames from the provided video, overlays MediaPipe pose landmarks plus connective lines, and writes a looping GIF. Adjust `--stride` to skip frames (reduces GIF size) and `--resize-width` to downscale while preserving aspect ratio.

7. Capture or convert skeleton-only visuals:

```powershell
# Capture 8 seconds from the default webcam and save a skeleton MP4
python -m src.pose_recorder camera -o skeleton.mp4 --duration 8

# Convert an existing video into a skeleton-only GIF (every other frame)
python -m src.pose_recorder path\to\clip.mp4 -o skeleton.gif --stride 2

# Pull a YouTube clip and export its skeleton overlay
python -m src.pose_recorder https://www.youtube.com/watch?v=dQw4w9WgXcQ -o skeleton.gif --stride 3
```

The recorder renders a clean pose skeleton on a configurable background without the original pixels. Use `--format mp4` to force MP4 output, `--background` to set a hex color (e.g. `#101830`), and `--fps` to control playback speed.

## Features

- Pose detection powered by MediaPipe with exponential smoothing for stable landmarks.
- IoU-based tracker to keep bounding boxes steady.
- Sport heuristics (push-up, squat, jump) with automatic rep detection and noise gating.
- Animated SVG demos that mirror the expected motion.
- Command-line video annotator that exports looping pose GIFs (local files or direct YouTube URLs).
- Skeleton-only recorder for webcams and existing videos (GIF or MP4 output).
- Session recording, JSONL export, optional Google Sheets summary upload, and a Chart.js viewer with rep overlays.

## Google Sheets

Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of a service account JSON and pass `SHEETS_ID` as an env var to enable uploads when a recording stops.

## Next steps

- Extend sport library (lunges, sit-ups, yoga poses) with tailored heuristics.
- Enrich UI with multi-athlete support and mobile layout polishing.
- Automate CI (lint + pytest) and ship a packaged desktop/web release.
