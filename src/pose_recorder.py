"""Capture pose skeletons from a camera or video file and export as GIF or MP4.

Examples (PowerShell):
    # Capture 8 seconds from the default webcam and save as MP4
    python -m src.pose_recorder camera -o skeleton.mp4 --duration 8

    # Convert an existing video into a skeleton-only GIF
    python -m src.pose_recorder path\to\clip.mp4 -o skeleton.gif --stride 2
"""
from __future__ import annotations

import argparse
import contextlib
import math
import re
import tempfile
import time
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple
from urllib.parse import urlparse

import cv2
import imageio  # type: ignore
import numpy as np

from .pose_detector import PoseDetector

VISIBILITY_THRESHOLD = 0.35
BACKGROUND_RGB = (12, 16, 32)

try:  # mediapipe >= 0.10 ships the canonical package name
    import mediapipe as mp

    pose_module = mp.solutions.pose  # type: ignore[attr-defined]
except AttributeError:  # fallback for legacy layout
    from mediapipe.python import solutions as mp_solutions

    pose_module = mp_solutions.pose

POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = tuple(
    (
        int(getattr(a, "value", a)),
        int(getattr(b, "value", b)),
    )
    for a, b in pose_module.POSE_CONNECTIONS
)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    return slug or "capture"


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


@contextlib.contextmanager
def _ensure_local_source(source: str) -> Iterator[Tuple[Path, str]]:
    if not _is_http_url(source):
        path = Path(source)
        yield path, path.stem
        return

    try:
        from yt_dlp import YoutubeDL  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "yt-dlp is required to download videos from URLs. Install it via 'pip install yt-dlp'."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="pose_recorder_") as tmpdir:
        tmp_path = Path(tmpdir)
        ydl_opts = {
            "outtmpl": str(tmp_path / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "format": "bv*+ba/best",
            "merge_output_format": "mp4",
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=True)
                if info is None:
                    raise RuntimeError("No media information returned")
                if "entries" in info:
                    info = info["entries"][0]
                downloaded = Path(ydl.prepare_filename(info))
                suggested = info.get("title") or info.get("id") or downloaded.stem
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            hint = "Ensure ffmpeg is installed and yt-dlp is up to date."
            raise SystemExit(
                f"Failed to download video (yt-dlp exited with code {code}). {hint}"
            ) from None
        except Exception as exc:
            raise SystemExit(f"Failed to download video: {exc}") from exc

        if not downloaded.exists():
            candidates = list(tmp_path.glob("*"))
            if not candidates:
                raise SystemExit("Video download completed but no output file was found.")
            downloaded = candidates[0]
            suggested = downloaded.stem

        yield downloaded, suggested


def _parse_background(value: Optional[str]) -> Tuple[int, int, int]:
    if not value:
        return BACKGROUND_RGB
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) not in (3, 6):
        raise ValueError("background color must be a 3 or 6 character hex string")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
    except ValueError as exc:
        raise ValueError("invalid hex color") from exc
    return (r, g, b)


def _draw_skeleton(canvas: np.ndarray, landmarks) -> None:
    """Draw pose landmarks on the provided canvas (BGR in-place)."""
    if landmarks is None:
        return
    h, w = canvas.shape[:2]
    points = []
    for lm in landmarks.landmark:
        if lm.visibility < VISIBILITY_THRESHOLD:
            points.append(None)
            continue
        x = int(lm.x * w)
        y = int(lm.y * h)
        if x < 0 or y < 0 or x >= w or y >= h:
            points.append(None)
            continue
        points.append((x, y))

    for start_idx, end_idx in POSE_CONNECTIONS:
        start = points[start_idx] if start_idx < len(points) else None
        end = points[end_idx] if end_idx < len(points) else None
        if start and end:
            cv2.line(canvas, start, end, (0, 255, 200), 3, lineType=cv2.LINE_AA)

    for pt in points:
        if pt:
            cv2.circle(canvas, pt, 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


def _determine_output_format(path: Path, override: Optional[str]) -> str:
    if override:
        return override.lower()
    suffix = path.suffix.lower()
    if suffix == ".gif":
        return "gif"
    return "mp4"


def _target_fps(cap: cv2.VideoCapture, user_fps: Optional[float], *, camera: bool) -> float:
    if user_fps and user_fps > 0:
        return float(user_fps)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not camera and fps > 0:
        return fps
    return 15.0


def _frame_size(frame: np.ndarray, resize_width: Optional[int]) -> Tuple[int, int]:
    if resize_width and frame.shape[1] > resize_width:
        scale = resize_width / frame.shape[1]
        h = int(math.ceil(frame.shape[0] * scale))
        return resize_width, h
    return frame.shape[1], frame.shape[0]


def _process_frames(
    cap: cv2.VideoCapture,
    *,
    camera: bool,
    duration: float,
    max_frames: Optional[int],
    stride: int,
    resize_width: Optional[int],
    output_path: Path,
    output_format: str,
    fps: float,
    background: Tuple[int, int, int],
) -> Path:
    if not cap.isOpened():
        raise RuntimeError("Could not open source video/camera")

    detector = PoseDetector(model_complexity=1, smooth=True)

    writer: Optional[cv2.VideoWriter] = None
    gif_writer = None

    frame_index = 0
    processed = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            if stride > 1 and (frame_index - 1) % stride != 0:
                continue

            if camera:
                if duration > 0 and (time.time() - start_time) >= duration:
                    break
            elif max_frames and processed >= max_frames:
                break

            width, height = _frame_size(frame, resize_width)
            if (width, height) != (frame.shape[1], frame.shape[0]):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, landmarks = detector.detect(rgb)

            canvas = np.zeros_like(frame)
            canvas[:, :] = background[::-1]  # convert RGB -> BGR for OpenCV drawing
            _draw_skeleton(canvas, landmarks)

            if output_format == "gif":
                if gif_writer is None:
                    frame_duration = 1.0 / fps if fps > 0 else 1.0 / 15.0
                    gif_writer = imageio.get_writer(
                        output_path,
                        mode="I",
                        duration=frame_duration,
                        loop=0,
                    )
                gif_writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            else:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas.shape[1], canvas.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("Could not open video writer")
                writer.write(canvas)

            processed += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
        detector.close()
        if writer is not None:
            writer.release()
        if gif_writer is not None:
            gif_writer.close()

    if processed == 0:
        raise RuntimeError("No frames were processed; check camera availability or stride/max-frames settings.")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture pose skeletons and export as GIF/MP4")
    parser.add_argument("source", help="'camera' to use the default webcam, or path/URL to a video file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file path (.gif or .mp4)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index when using live capture")
    parser.add_argument("--duration", type=float, default=10.0, help="Capture duration in seconds when using the camera")
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames from the source video")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame to reduce output size")
    parser.add_argument("--resize-width", type=int, default=None, help="Resize frames to this width while keeping aspect ratio")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS for output (GIF playback or MP4 encoding)")
    parser.add_argument("--format", choices=["gif", "mp4"], help="Override output format (otherwise deduced from extension)")
    parser.add_argument("--background", type=str, default=None, help="Hex RGB background color, e.g. #0a1020")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be ≥ 1")

    background = _parse_background(args.background)

    if args.source.lower() == "camera" and args.duration <= 0:
        raise SystemExit("--duration must be positive when capturing from the camera")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_format = _determine_output_format(output_path, args.format)

    if args.source.lower() == "camera":
        cap = cv2.VideoCapture(args.camera_index)
        fps = _target_fps(cap, args.fps, camera=True)
        try:
            result = _process_frames(
                cap,
                camera=True,
                duration=args.duration,
                max_frames=args.max_frames,
                stride=args.stride,
                resize_width=args.resize_width,
                output_path=output_path,
                output_format=output_format,
                fps=fps,
                background=background,
            )
        except Exception as exc:
            raise SystemExit(f"Failed to record skeletons: {exc}") from exc
    else:
        with _ensure_local_source(args.source) as (local_path, _):
            cap = cv2.VideoCapture(str(local_path))
            fps = _target_fps(cap, args.fps, camera=False)
            try:
                result = _process_frames(
                    cap,
                    camera=False,
                    duration=args.duration,
                    max_frames=args.max_frames,
                    stride=args.stride,
                    resize_width=args.resize_width,
                    output_path=output_path,
                    output_format=output_format,
                    fps=fps,
                    background=background,
                )
            except Exception as exc:
                raise SystemExit(f"Failed to record skeletons: {exc}") from exc

    print(f"Saved skeleton overlay to {result}")


if __name__ == "__main__":
    main()
