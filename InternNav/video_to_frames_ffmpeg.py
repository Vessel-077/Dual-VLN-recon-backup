#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_to_frames_ffmpeg.py

Extract frames from a video at a given resolution and frame rate,
saving them as a sequence of images. This version drives the ffmpeg
command-line tool via subprocess instead of using OpenCV.

The input video is specified as a directory + a file name, so that on
the command line you usually only need to pass --video-name (the
directory falls back to DEFAULT_VIDEO_DIR).

Dependencies:
    - The ffmpeg command-line tool must be installed and on PATH
      (see the install instructions provided separately).
    - No extra Python packages are required (only the standard library).

Usage examples:
    # Use the built-in defaults for everything (video dir + video name)
    python video_to_frames_ffmpeg.py

    # Only specify the video file name; directory/resolution/fps/etc. use defaults
    python video_to_frames_ffmpeg.py --video-name my_clip.mp4

    # Fully custom
    python video_to_frames_ffmpeg.py \
        --video-dir "/path/to/videos" \
        --video-name "my_clip.mp4" \
        --outdir "/path/to/output_frames" \
        --width 1280 --height 720 \
        --fps 6 \
        --prefix debug_raw \
        --ext jpg
"""

import argparse
import os
import shutil
import subprocess
import sys


# ==========================================================
# Default configuration: edit directly here, or override via CLI args
# ==========================================================
DEFAULT_VIDEO_DIR = "assets/videos"                # Default directory that holds the input video
DEFAULT_VIDEO_NAME = "demo.mp4"                      # Default input video file name (inside DEFAULT_VIDEO_DIR)
DEFAULT_OUTPUT_DIR = "assets/open_loop_data/sample_01"        # Default output image directory (edit as needed)
DEFAULT_WIDTH = 1280                                 # Default output image width
DEFAULT_HEIGHT = 720                                 # Default output image height
DEFAULT_FPS = 6.0                                    # Default sampling frame rate (frames extracted per second)
DEFAULT_PREFIX = "debug_raw"                         # Output image filename prefix
DEFAULT_EXT = "jpg"                                  # Output image format: png / jpg etc.
DEFAULT_DIGITS = 4                                   # Number of digits in the filename index, e.g. 0000
DEFAULT_JPG_QUALITY = 2                              # ffmpeg -q:v value for jpg (1=best, 31=worst)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose a video into a sequence of images at a given resolution and "
                    "frame rate, using the ffmpeg command-line tool."
    )
    parser.add_argument(
        "--video-dir", "-d",
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help=f"Directory that holds the input video (default: {DEFAULT_VIDEO_DIR})",
    )
    parser.add_argument(
        "--video-name", "-n",
        type=str,
        default=DEFAULT_VIDEO_NAME,
        help=f"Input video file name, relative to --video-dir (default: {DEFAULT_VIDEO_NAME}). "
             f"You may also pass an absolute path here to override --video-dir entirely.",
    )
    parser.add_argument(
        "--outdir", "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output image directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Output image width (default: {DEFAULT_WIDTH}); set to 0 to keep the original width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Output image height (default: {DEFAULT_HEIGHT}); set to 0 to keep the original height",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Sampling frame rate, i.e. how many images to extract per second (default: {DEFAULT_FPS}). "
             f"Set to 0 or negative to extract every single frame from the video (no fps filter applied).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help=f"Output image filename prefix (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=DEFAULT_EXT,
        choices=["png", "jpg", "jpeg", "bmp", "webp"],
        help=f"Output image format (default: {DEFAULT_EXT})",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=DEFAULT_DIGITS,
        help=f"Number of digits in the filename index (default: {DEFAULT_DIGITS}), e.g. 4 -> debug_raw_0000.jpg",
    )
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help="Preserve the original aspect ratio when resizing to the target resolution "
             "(scale to cover, then center-crop to fill the target size). "
             "If not set, the frame is stretched directly to width x height.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=DEFAULT_JPG_QUALITY,
        help=f"ffmpeg JPEG quality (-q:v), 1 (best) to 31 (worst); only used when --ext is jpg/jpeg "
             f"(default: {DEFAULT_JPG_QUALITY})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory without asking (adds ffmpeg -y).",
    )
    return parser.parse_args()


def resolve_video_path(video_dir, video_name):
    """Combine --video-dir and --video-name into a single path.
    If video_name is already an absolute path, it is used as-is."""
    if os.path.isabs(video_name):
        return video_name
    return os.path.join(video_dir, video_name)


def check_ffmpeg_available():
    """Make sure the ffmpeg binary can be found on PATH."""
    if shutil.which("ffmpeg") is None:
        print(
            "[Error] 'ffmpeg' was not found on PATH. Please install it first "
            "(see the install instructions), then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def build_video_filter(width, height, fps, keep_aspect):
    """Build the ffmpeg -vf filter string for fps sampling and resizing."""
    filters = []

    # --- frame sampling ---
    if fps > 0:
        filters.append(f"fps={fps}")

    # --- resizing ---
    if width > 0 or height > 0:
        # 0 means "keep this dimension proportional / unchanged" -> ffmpeg uses -1
        w = width if width > 0 else -1
        h = height if height > 0 else -1

        if keep_aspect and width > 0 and height > 0:
            # Scale to cover the target box, then center-crop to exactly width x height
            filters.append(
                f"scale=w={width}:h={height}:force_original_aspect_ratio=increase"
            )
            filters.append(f"crop={width}:{height}")
        else:
            # Direct stretch to width x height (or proportional scale if one side is 0)
            filters.append(f"scale={w}:{h}")

    return ",".join(filters) if filters else None


def main():
    args = parse_args()

    check_ffmpeg_available()

    video_path = resolve_video_path(args.video_dir, args.video_name)

    if not os.path.isfile(video_path):
        print(f"[Error] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    vf = build_video_filter(args.width, args.height, args.fps, args.keep_aspect)

    output_pattern = os.path.join(
        args.outdir, f"{args.prefix}_%0{args.digits}d.{args.ext}"
    )

    cmd = ["ffmpeg"]
    cmd += ["-y"] if args.overwrite else ["-n"]
    cmd += ["-i", video_path]
    if vf:
        cmd += ["-vf", vf]
    if args.ext in ("jpg", "jpeg"):
        cmd += ["-q:v", str(args.jpg_quality)]
    cmd += ["-start_number", "0", output_pattern]

    print("Running command:")
    print(" ".join(cmd))
    print(f"Input video: {video_path}")
    print(f"Output directory: {args.outdir}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[Error] ffmpeg exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    saved = [
        f for f in os.listdir(args.outdir)
        if f.startswith(args.prefix) and f.endswith(f".{args.ext}")
    ]
    print(f"Done! {len(saved)} images are now in: {args.outdir}")


if __name__ == "__main__":
    main()