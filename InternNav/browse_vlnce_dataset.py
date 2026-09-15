#!/usr/bin/env python
"""
Quickly browse InternNav's VLN-CE (R2R) annotation data: instructions, scene ids,
episode counts per split, and a quick 2D top-down plot of each episode's reference
path (from the raw coordinates in the JSON — no simulator/scene rendering needed).

Expects the InternData-N1 vln_ce layout:
  data/vln_ce/raw_data/r2r/{train,val_seen,val_unseen}/*.json.gz

Run from the InternNav repo root:
    python browse_vlnce_dataset.py --split val_unseen
Outputs go to ./vlnce_overview/<split>/
"""
import argparse
import csv
import glob
import gzip
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = "data/vln_ce/raw_data/r2r"
OUT_ROOT = "vlnce_overview"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=None,
                    help="e.g. val_unseen / val_seen / train. Default: all splits found.")
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--max-table-rows", type=int, default=200,
                    help="cap rows written to the markdown preview table (CSV always gets everything)")
    p.add_argument("--plot-n", type=int, default=16,
                    help="how many episodes' reference paths to plot in the contact-sheet PNG")
    return p.parse_args()


def find_split_files(data_root, split):
    pattern = os.path.join(data_root, split, "*.json.gz") if split else os.path.join(data_root, "*", "*.json.gz")
    return sorted(glob.glob(pattern))


def load_episodes(json_gz_path):
    with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    keys = list(data.keys()) if isinstance(data, dict) else type(data)
    raise ValueError(f"Unrecognized JSON structure in {json_gz_path}: top-level = {keys}")


def get_instruction_text(ep):
    instr = ep.get("instruction")
    if isinstance(instr, dict):
        return instr.get("instruction_text") or str(instr)
    if isinstance(instr, list):
        texts = [it.get("instruction_text", str(it)) if isinstance(it, dict) else str(it) for it in instr]
        return " | ".join(texts)
    if isinstance(instr, str):
        return instr
    return ""


def get_path(ep):
    for key in ("reference_path", "path", "gt_path"):
        if key in ep and ep[key]:
            return ep[key]
    return None


def main():
    args = parse_args()
    files = find_split_files(args.data_root, args.split)
    if not files:
        print(f"No *.json.gz files found under {args.data_root} (split={args.split!r}). "
              f"Check the path / whether the dataset has been downloaded yet.")
        return

    for json_gz_path in files:
        #split_name = os.path.basename(os.path.dirname(json_gz_path))
        split_name = os.path.basename(json_gz_path)
        for suffix in (".json.gz", ".gz", ".json"):
            if split_name.endswith(suffix):
                split_name = split_name[: -len(suffix)]
                break
        out_dir = os.path.join(OUT_ROOT, split_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== Loading {json_gz_path} ===")
        episodes = load_episodes(json_gz_path)
        print(f"{len(episodes)} episodes")
        if episodes:
            print("Keys in first episode:", sorted(episodes[0].keys()))

        csv_path = os.path.join(out_dir, "episodes_summary.csv")
        rows = []
        scenes = set()
        for ep in episodes:
            scene_id = ep.get("scene_id", "")
            scenes.add(scene_id)
            instruction = get_instruction_text(ep)
            path = get_path(ep)
            geo_dist = None
            info = ep.get("info")
            if isinstance(info, dict):
                geo_dist = info.get("geodesic_distance")
            rows.append({
                "episode_id": ep.get("episode_id", ""),
                "trajectory_id": ep.get("trajectory_id", ""),
                "scene_id": scene_id,
                "num_waypoints": len(path) if path else "",
                "geodesic_distance": geo_dist if geo_dist is not None else "",
                "instruction": instruction,
            })

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode_id", "trajectory_id", "scene_id",
                                                     "num_waypoints", "geodesic_distance", "instruction"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV summary ({len(rows)} rows) -> {csv_path}")

        md_path = os.path.join(out_dir, "SUMMARY.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {split_name} — {len(episodes)} episodes, {len(scenes)} unique scenes\n\n")
            f.write("| episode_id | scene_id | waypoints | geodesic_dist | instruction |\n")
            f.write("|---|---|---|---|---|\n")
            for row in rows[: args.max_table_rows]:
                instr_short = (row["instruction"][:100] + "...") if len(row["instruction"]) > 100 else row["instruction"]
                f.write(f"| {row['episode_id']} | {row['scene_id']} | {row['num_waypoints']} | "
                        f"{row['geodesic_distance']} | {instr_short} |\n")
            if len(rows) > args.max_table_rows:
                f.write(f"\n... ({len(rows) - args.max_table_rows} more rows in episodes_summary.csv)\n")
        print(f"Markdown preview -> {md_path}")

        eps_with_path = [ep for ep in episodes if get_path(ep)][: args.plot_n]
        if eps_with_path:
            cols = 4
            rows_n = (len(eps_with_path) + cols - 1) // cols
            fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3, rows_n * 3))
            axes = np.array(axes).reshape(-1)
            for ax, ep in zip(axes, eps_with_path):
                path = np.array(get_path(ep))
                x = path[:, 0]
                z = path[:, 2] if path.shape[1] > 2 else path[:, 1]
                ax.plot(x, z, "b-", linewidth=1.5)
                ax.plot(x[0], z[0], "go", markersize=5)
                ax.plot(x[-1], z[-1], "ro", markersize=5)
                ax.set_title(f"ep {ep.get('episode_id', '?')}", fontsize=7)
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=5)
            for ax in axes[len(eps_with_path):]:
                ax.axis("off")
            fig.suptitle(f"{split_name}: top-down reference paths (first {len(eps_with_path)} episodes)", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            png_path = os.path.join(out_dir, "reference_paths_contact_sheet.png")
            fig.savefig(png_path, dpi=130)
            plt.close(fig)
            print(f"Reference-path contact sheet -> {png_path}")
        else:
            print("No episodes with a usable reference/path field found for plotting.")


if __name__ == "__main__":
    main()
