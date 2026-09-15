#!/usr/bin/env python
"""
Scan vlnce_overview/*/episodes_summary.csv (from browse_vlnce_dataset.py) for
episodes whose instruction contains a keyword (case-insensitive, default "turn
around"), and write a manifest of [scene_id, episode_id] pairs that
internnav/env/habitat_env.py's patched generate_episodes() filters on via the
INTERNNAV_EPISODE_FILTER env var.

Run from the InternNav repo root:
    python select_turn_around_episodes.py --keyword "turn around"
"""
import argparse
import csv
import glob
import json
import os


def normalize_scene_id(raw_scene_id):
    parts = raw_scene_id.split('/')
    return parts[-2] if len(parts) >= 2 else raw_scene_id


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--keyword", default="turn around")
    p.add_argument("--overview-root", default="vlnce_overview")
    p.add_argument("--out", default="turn_around_episodes.json")
    return p.parse_args()


def main():
    args = parse_args()
    keyword = args.keyword.lower()

    matches = []
    csv_paths = sorted(glob.glob(os.path.join(args.overview_root, "*", "episodes_summary.csv")))
    if not csv_paths:
        print(f"No episodes_summary.csv found under {args.overview_root}/*/ - run browse_vlnce_dataset.py first?")
        return

    for csv_path in csv_paths:
        split_name = os.path.basename(os.path.dirname(csv_path))
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                instruction = row.get("instruction", "") or ""
                if keyword in instruction.lower():
                    scene_id = normalize_scene_id(row.get("scene_id", ""))
                    episode_id = row.get("episode_id", "")
                    matches.append({
                        "split": split_name,
                        "scene_id": scene_id,
                        "episode_id": int(episode_id) if str(episode_id).strip() else None,
                        "instruction": instruction,
                    })

    print(f"Found {len(matches)} episodes containing {args.keyword!r} across {len(csv_paths)} CSV file(s):")
    for m in matches:
        print(f"  [{m['split']}] scene={m['scene_id']} episode_id={m['episode_id']}: {m['instruction'][:80]}")

    manifest = [[m["scene_id"], m["episode_id"]] for m in matches if m["episode_id"] is not None]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote filter manifest ({len(manifest)} entries) -> {args.out}")

    readable_out = os.path.splitext(args.out)[0] + "_readable.json"
    with open(readable_out, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)
    print(f"Human-readable copy (with split + full instruction) -> {readable_out}")


if __name__ == "__main__":
    main()
