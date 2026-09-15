#!/usr/bin/env python
"""
Quickly browse InternNav real-world sample scenes: for each scene folder under
assets/, produce a contact-sheet PNG (sampled frames + instruction text) and a
low-res preview GIF, plus a text summary of all instructions.

Run from the InternNav repo root:
    python browse_dataset.py
Outputs go to ./dataset_overview/
"""
import glob
import os
import re

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ASSETS_DIR = "assets"
OUT_DIR = "dataset_overview"
N_THUMBS = 12          # frames per contact sheet
GIF_MAX_FRAMES = 60    # cap gif length for speed
GIF_SIZE = (256, 192)
GIF_FPS = 8


def find_instruction_file(scene_dir):
    for name in ("instruction.txt", "insturction.txt"):  # some scenes ship a typo'd filename
        p = os.path.join(scene_dir, name)
        if os.path.exists(p):
            return p
    return None


def natural_key(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return [int(n) for n in nums] if nums else [0]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scene_dirs = sorted(d for d in glob.glob(os.path.join(ASSETS_DIR, "*")) if os.path.isdir(d))

    summary_lines = []

    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        instr_path = find_instruction_file(scene_dir)
        rgb_paths = sorted(
            (p for p in glob.glob(os.path.join(scene_dir, "debug_raw_*.jpg")) if "look_down" not in p),
            key=natural_key,
        )
        if not rgb_paths:
            continue

        instruction = ""
        if instr_path:
            with open(instr_path, "r") as f:
                instruction = f.read().strip()

        n_frames = len(rgb_paths)
        n_look_down = len(glob.glob(os.path.join(scene_dir, "debug_raw_*_look_down.jpg")))

        summary_lines.append(f"## {scene_name}")
        summary_lines.append(f"- frames: {n_frames} (+ {n_look_down} look_down)")
        summary_lines.append(f"- instruction: {instruction if instruction else '(instruction.txt NOT FOUND)'}")
        summary_lines.append("")

        # ---- contact sheet ----
        idxs = np.linspace(0, n_frames - 1, min(N_THUMBS, n_frames)).astype(int)
        idxs = sorted(set(idxs.tolist()))
        cols = 4
        rows = (len(idxs) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.4))
        axes = np.array(axes).reshape(-1)
        for ax, idx in zip(axes, idxs):
            img = Image.open(rgb_paths[idx]).convert("RGB")
            ax.imshow(img)
            ax.set_title(os.path.basename(rgb_paths[idx]), fontsize=7)
            ax.axis("off")
        for ax in axes[len(idxs):]:
            ax.axis("off")
        wrapped = "\n".join(instruction[i:i + 90] for i in range(0, len(instruction), 90)) if instruction else "(no instruction.txt)"
        fig.suptitle(f"{scene_name}  ({n_frames} frames)\n{wrapped}", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out_png = os.path.join(OUT_DIR, f"{scene_name}_contact_sheet.png")
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        print(f"[{scene_name}] contact sheet -> {out_png}")

        # ---- preview gif ----
        gif_paths = rgb_paths[:: max(1, n_frames // GIF_MAX_FRAMES)]
        frames = []
        for p in gif_paths:
            img = Image.open(p).convert("RGB").resize(GIF_SIZE)
            frames.append(np.array(img))
        out_gif = os.path.join(OUT_DIR, f"{scene_name}_preview.gif")
        imageio.mimsave(out_gif, frames, fps=GIF_FPS)
        print(f"[{scene_name}] preview gif  -> {out_gif} ({len(frames)} frames)")

    summary_path = os.path.join(OUT_DIR, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write("# Dataset scene overview\n\n")
        f.write("\n".join(summary_lines))
    print(f"\nSummary written -> {summary_path}")


if __name__ == "__main__":
    main()