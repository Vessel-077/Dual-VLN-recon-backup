#!/usr/bin/env python
"""
Dual-System VLN (InternVLA-N1-DualVLN) offline inference -> annotated frames -> video.

Adapted from InternRobotics/InternNav's official notebook:
  scripts/notebooks/inference_only_demo.ipynb
Run this from inside <InternNav repo root>/scripts/notebooks/, exactly where the
notebook itself expects to be run from (it uses ../../ relative paths for the repo
root, assets, and checkpoints).

What it adds on top of the notebook:
  - CLI args instead of hard-coded personal paths
  - annotates every frame (not only the ones with a trajectory) so the output
    image sequence has one frame per input frame, in order
  - stitches the annotated frames into an .mp4 (and optionally a .gif preview)
    at the end using imageio, which the notebook does not do itself
  - a safe font fallback if DejaVuSansMono.ttf isn't installed on the system
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser(description="Run InternVLA-N1-DualVLN offline inference demo and export a video.")
    p.add_argument("--repo-root", type=str, default="../../",
                    help="Path to the InternNav repo root (default: ../../, i.e. run from scripts/notebooks/)")
    p.add_argument("--model-path", type=str, required=True,
                    help="Path to the downloaded InternVLA-N1-DualVLN checkpoint directory")
    p.add_argument("--scene-dir", type=str, default=None,
                    help="Path to a scene folder with debug_raw_*.jpg + instruction.txt "
                         "(default: <repo-root>/assets/realworld_sample_data1)")
    p.add_argument("--save-dir", type=str, default="./demo_out",
                    help="Where to write annotated frames and the final video")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resize-w", type=int, default=384)
    p.add_argument("--resize-h", type=int, default=384)
    p.add_argument("--num-history", type=int, default=8)
    p.add_argument("--plan-step-gap", type=int, default=4,
                    help="Run S2 planning every N frames (matches the notebook default)")
    p.add_argument("--fps", type=int, default=6, help="FPS for the output video")
    p.add_argument("--gif", action="store_true", help="Also write a .gif preview alongside the .mp4")
    p.add_argument("--no-flash-attn", action="store_true",
                    help="Set this if flash-attn isn't installed / doesn't match your torch+CUDA build")
    return p.parse_args()


def annotate_image(idx_str, image, llm_output, trajectory, pixel_goal, output_dir):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font_size = 20
    font = None
    for candidate in (
        "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    text_content = [f"Frame    Id  : {idx_str}", f"Actions      : {llm_output}"]
    max_width = 0
    total_height = 0
    for line in text_content:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_width = max(max_width, bbox[2] - bbox[0])
        total_height += 26

    padding = 10
    box_x, box_y = 10, 10
    box_width = max_width + 2 * padding
    box_height = total_height + 2 * padding
    draw.rectangle([box_x, box_y, box_x + box_width, box_y + box_height], fill="black")

    y_position = box_y + padding
    for line in text_content:
        draw.text((box_x + padding, y_position), line, fill="white", font=font)
        y_position += 26

    image = np.array(image)

    if trajectory is not None and len(trajectory) > 0:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        img_height, img_width = image.shape[:2]
        window_size = 200
        window_x = img_width - window_size
        window_y = 0

        traj_points = [[float(pt[0]), float(pt[1])] for pt in trajectory
                       if isinstance(pt, (list, tuple, np.ndarray)) and len(pt) >= 2]

        if len(traj_points) > 0:
            traj_array = np.array(traj_points)
            x_coords, y_coords = traj_array[:, 0], traj_array[:, 1]

            fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
            fig.patch.set_alpha(0.6)
            fig.patch.set_facecolor("gray")
            ax.set_facecolor("lightgray")

            ax.plot(y_coords, x_coords, "b-", linewidth=2, label="Trajectory")
            ax.plot(y_coords[0], x_coords[0], "go", markersize=6, label="Start")
            ax.plot(y_coords[-1], x_coords[-1], "ro", markersize=6, label="End")
            ax.plot(0, 0, "w+", markersize=10, markeredgewidth=2, label="Origin")

            ax.set_xlabel("Y (left +)", fontsize=8)
            ax.set_ylabel("X (up +)", fontsize=8)
            ax.invert_xaxis()
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_aspect("equal", adjustable="box")
            ax.legend(fontsize=6, loc="upper right")
            plt.tight_layout(pad=0.3)

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            plot_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            plot_img = cv2.resize(plot_img, (window_size, window_size))
            image[window_y:window_y + window_size, window_x:window_x + window_size] = plot_img

    if pixel_goal is not None:
        cv2.circle(image, (int(pixel_goal[1]), int(pixel_goal[0])), 5, (255, 0, 0), -1)

    image = Image.fromarray(image).convert("RGB")
    out_path = os.path.join(output_dir, f"frame_{idx_str}_annotated.png")
    image.save(out_path)
    return out_path


def main():
    args = parse_args()

    if args.no_flash_attn:
        os.environ["INTERNNAV_DISABLE_FLASH_ATTN"] = "1"  # harmless if the agent code ignores it

    repo_root = Path(args.repo_root)
    sys.path.insert(0, str(repo_root))
    # NOTE: the official notebook says 'src/diffusion-policy', but the actual submodule
    # path in the current repo (per .gitmodules) is 'third_party/diffusion-policy'.
    sys.path.insert(0, str(repo_root / "third_party/diffusion-policy"))

    from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

    class Args:
        pass

    a = Args()
    a.device = args.device
    a.model_path = args.model_path
    a.resize_w = args.resize_w
    a.resize_h = args.resize_h
    a.num_history = args.num_history
    a.camera_intrinsic = np.array([
        [386.5, 0.0, 328.9, 0.0],
        [0.0, 386.5, 244.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    a.plan_step_gap = args.plan_step_gap

    print(f"Model path : {a.model_path}")
    print(f"Device     : {a.device}")
    print(f"Image size : {a.resize_w}x{a.resize_h}")

    print("Loading model...")
    agent = InternVLAN1AsyncAgent(a)

    print("Warming up model...")
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.float32)
    dummy_pose = np.eye(4)
    agent.reset()
    agent.step(dummy_rgb, dummy_depth, dummy_pose, "hello", intrinsic=a.camera_intrinsic)
    print("Model loaded successfully!")

    scene_dir = args.scene_dir or str(repo_root / "assets" / "realworld_sample_data1")
    instruction_path = os.path.join(scene_dir, "instruction.txt")
    if not os.path.exists(instruction_path):
        raise FileNotFoundError(f"instruction.txt not found in {scene_dir}")

    with open(instruction_path, "r") as f:
        instruction = f.read().strip()

    rgb_paths = sorted(glob.glob(os.path.join(scene_dir, "debug_raw_*.jpg")))
    print(f"Scene: {scene_dir}")
    print(f"Instruction: {instruction}")
    print(f"Found {len(rgb_paths)} images")
    if len(rgb_paths) == 0:
        raise RuntimeError("No debug_raw_*.jpg frames found - did you extract realworld_sample_data.tar.gz?")

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    agent.reset()
    print("=" * 80)
    print(f"Processing scene: {os.path.basename(scene_dir)}")
    print(f"Instruction: '{instruction}'")
    print(f"Total images: {len(rgb_paths)}")
    print("=" * 80)

    saved_frames = []
    for i, rgb_path in enumerate(rgb_paths):
        look_down = "look_down" in rgb_path
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))

        # Sample data has no recorded depth -> constant dummy depth, matching the
        # official notebook. Swap this for real aligned depth if you build your own dataset.
        depth = 10 * np.ones((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        camera_pose = np.eye(4)

        with torch.no_grad():
            out = agent.step(
                rgb, depth, camera_pose, instruction,
                intrinsic=a.camera_intrinsic, look_down=look_down,
            )

        idx_str = f"{i:04d}"
        trajectory = None
        pixel_goal = None
        action_text = "-"

        if out.output_action is not None and out.output_action != []:
            action_text = str(out.output_action)
            print(f"[{idx_str}] action: {out.output_action}")
        else:
            trajectory = out.output_trajectory.tolist() if out.output_trajectory is not None else None
            pixel_goal = out.output_pixel
            print(f"[{idx_str}] trajectory: {trajectory}  pixel_goal: {pixel_goal}")

        frame_path = annotate_image(idx_str, rgb, action_text, trajectory, pixel_goal, save_dir)
        saved_frames.append(frame_path)

    print(f"\nScene {os.path.basename(scene_dir)} completed! {len(saved_frames)} annotated frames in {save_dir}")

    # ---- Stitch frames into a video (this part is NOT in the official notebook) ----
    import imageio.v2 as imageio

    video_path = os.path.join(save_dir, "dualvln_demo.mp4")
    print(f"Writing video to {video_path} at {args.fps} fps ...")
    writer = imageio.get_writer(video_path, fps=args.fps, macro_block_size=None)
    for frame_path in saved_frames:
        writer.append_data(imageio.imread(frame_path))
    writer.close()
    print(f"Video written: {video_path}")

    if args.gif:
        gif_path = os.path.join(save_dir, "dualvln_demo.gif")
        print(f"Writing gif preview to {gif_path} ...")
        imgs = [imageio.imread(fp) for fp in saved_frames]
        imageio.mimsave(gif_path, imgs, fps=args.fps)
        print(f"GIF written: {gif_path}")


if __name__ == "__main__":
    main()