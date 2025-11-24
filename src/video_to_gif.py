"""Convert a recorded video into an annotated pose GIF.

Usage (PowerShell):
    python -m src.video_to_gif path\to\video.mp4 -o annotated.gif
"""
from __future__ import annotations

import argparse
import contextlib
import re
from pathlib import Path
import tempfile
from typing import Iterable, Iterator, Tuple
from urllib.parse import urlparse

import cv2
import imageio  # type: ignore
import numpy as np

from .pose_detector import PoseDetector

VISIBILITY_THRESHOLD = 0.35

try:  # mediapipe >= 0.10 ships the canonical package name
    import mediapipe as mp
    pose_module = mp.solutions.pose  # type: ignore[attr-defined]
except AttributeError:  # fallback for legacy layout
    from mediapipe.python import solutions as mp_solutions
    pose_module = mp_solutions.pose

POSE_CONNECTIONS: Iterable[Tuple[int, int]] = tuple(
    (
        int(getattr(a, "value", a)),
        int(getattr(b, "value", b)),
    )
    for a, b in pose_module.POSE_CONNECTIONS
)


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except Exception:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    return slug or "video"


@contextlib.contextmanager
def _resolve_video_source(source: str) -> Iterator[Tuple[Path, str]]:
    if _is_http_url(source):
        try:
            from yt_dlp import YoutubeDL  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SystemExit(
                "yt-dlp is required to download videos from URLs. Install it via 'pip install yt-dlp'."
            ) from exc

        with tempfile.TemporaryDirectory(prefix="video_to_gif_") as tmpdir:
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
                    if "entries" in info:  # playlist-like response
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
                # Some formats adjust extension during post-processing; pick any file in folder.
                candidates = list(tmp_path.glob("*"))
                if not candidates:
                    raise SystemExit("Video download completed but no output file was found.")
                downloaded = candidates[0]
                suggested = downloaded.stem

            yield downloaded, suggested
        return

    path = Path(source)
    yield path, path.stem


def draw_landmarks(image: np.ndarray, landmarks) -> None:
    if landmarks is None:
        return
    h, w = image.shape[:2]
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
            cv2.line(image, start, end, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    for pt in points:
        if pt:
            cv2.circle(image, pt, 3, (0, 191, 255), thickness=-1, lineType=cv2.LINE_AA)


def convert_video_to_gif(
    source: Path,
    *,
    output: Path,
    max_frames: int | None,
    stride: int,
    resize_width: int | None,
    output_fps: float | None,
) -> Path:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    # Determine playback FPS for the GIF. Respect explicit override, else inherit.
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    gif_fps = output_fps or (native_fps / stride if native_fps > 0 else 12.0)
    if gif_fps <= 0:
        gif_fps = 12.0

    detector = PoseDetector(model_complexity=1, smooth=True)
    gif_writer = None

    processed = 0
    grabbed = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            grabbed += 1

            if stride > 1:
                if (grabbed - 1) % stride != 0:
                    continue

            if resize_width and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                new_size = (resize_width, int(frame.shape[0] * scale))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, landmarks = detector.detect(rgb)
            draw_landmarks(frame, landmarks)

            if gif_writer is None:
                frame_duration = 1.0 / gif_fps if gif_fps > 0 else 1.0 / 12.0
                gif_writer = imageio.get_writer(
                    output,
                    mode="I",
                    duration=frame_duration,
                    loop=0,
                )

            gif_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            processed += 1

            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()
        detector.close()
        if gif_writer is not None:
            gif_writer.close()

    if processed == 0:
        raise RuntimeError("No frames were processed; check stride/max-frames settings.")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a video with pose landmarks and export as GIF.")
    parser.add_argument("video", type=str, help="Input video file path or HTTP(S) URL")
    parser.add_argument("-o", "--output", type=Path, help="Output GIF path (default: <video>.gif)")
    parser.add_argument("--max-frames", type=int, default=None, help="Process at most N frames")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--resize-width", type=int, default=None, help="Resize frames to this width while preserving aspect ratio")
    parser.add_argument("--fps", type=float, default=None, help="FPS for the output GIF (default: match input)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_arg: str = args.video

    if args.stride < 1:
        raise SystemExit("--stride must be ≥ 1")

    with _resolve_video_source(video_arg) as (resolved_video, suggested_name):
        if not resolved_video.exists():
            raise SystemExit(f"Input video not found: {resolved_video}")

        if args.output:
            output_path = args.output
        else:
            safe_name = _slugify(suggested_name)
            output_path = Path.cwd() / f"{safe_name}.gif"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = convert_video_to_gif(
                resolved_video,
                output=output_path,
                max_frames=args.max_frames,
                stride=args.stride,
                resize_width=args.resize_width,
                output_fps=args.fps,
            )
        except Exception as exc:
            raise SystemExit(f"Failed to convert video: {exc}") from exc

    print(f"Saved annotated GIF to {result}")


if __name__ == "__main__":
    main()
