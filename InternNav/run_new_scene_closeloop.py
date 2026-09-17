#!/usr/bin/env python
"""
All-in-one script: given a scene directory that contains only <scene_name>.glb,
automatically:
  1. Locate the scene-specific habitat yaml + eval config by scene name
     (must already be hand-prepared; raises an error if missing)
  2. Read the actually-effective agent radius/height and dataset paths from the yaml
  3. Check whether the navmesh exists; if not, bake it directly from the glb on the spot
     (never copies a navmesh from elsewhere)
  4. Generate 1 geometrically valid single-task episode for this scene, and write it to
     the data_path declared in the yaml
  5. Call scripts/eval/eval.py to run one real closed-loop inference pass, with the
     right environment variables set so that an annotated video + per-step annotated
     images are saved regardless of whether this task succeeds or fails
     (saving per-step images requires the patch in section 6 first; otherwise you
     only get the video, not the images)

Usage:
    python run_new_scene_closed_loop.py --scene-dir data/other_scene/mini/random_scene_1
"""
import argparse
import gzip
import json
import math
import os
import random
import subprocess
import sys

import habitat_sim
from habitat_baselines.config.default import get_config as get_habitat_config


def yaw_quat(angle_rad):
    return [0.0, math.sin(angle_rad / 2), 0.0, math.cos(angle_rad / 2)]


def sample_single_episode(pathfinder, min_dist, max_dist, max_tries=300):
    for _ in range(max_tries):
        p0 = pathfinder.get_random_navigable_point()
        p1 = pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath()
        path.requested_start = p0
        path.requested_end = p1
        if not pathfinder.find_path(path):
            continue
        if min_dist <= path.geodesic_distance <= max_dist:
            return p0, p1, path
    raise RuntimeError(
        "Failed to sample a start/goal pair within the distance bounds on this navmesh after many tries; "
        "the scene may be too small or split into disconnected islands, try widening --min-dist/--max-dist"
    )


def ensure_navmesh(glb_path, navmesh_path, agent_radius, agent_height):
    if os.path.exists(navmesh_path):
        print(f"[navmesh] already exists, reusing it: {navmesh_path}")
        return
    print(f"[navmesh] not found, baking on the spot from {glb_path} (radius={agent_radius}, height={agent_height}) ...")
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = glb_path
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()  # without this call, the two fields below are ineffective and default to 0
        settings.agent_radius = agent_radius
        settings.agent_height = agent_height
        ok = sim.recompute_navmesh(sim.pathfinder, settings)
        assert ok, "navmesh baking failed; check the glb's coordinate system (needs Y-up) / units"
        area = sim.pathfinder.navigable_area
        assert area > 0, "baked navigable area is 0; the scene may have no valid ground"
        sim.pathfinder.save_nav_mesh(navmesh_path)
    finally:
        sim.close()
    print(f"[navmesh] saved: {navmesh_path} (navigable area {area:.2f} m^2)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True,
                     help="e.g. data/other_scene/mini/random_scene_1; "
                          "the directory name is the scene name, and <scene_name>.glb must already be inside it")
    ap.add_argument("--min-dist", type=float, default=2.0)
    ap.add_argument("--max-dist", type=float, default=15.0)
    ap.add_argument("--instruction", default="Turn right and go forward. Stop by the third chair from left.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-eval", action="store_true",
                     help="stop after generating the episode; don't actually call eval.py (useful for debugging scene setup in isolation)")
    args = ap.parse_args()
    random.seed(args.seed)

    scene_dir = args.scene_dir.rstrip("/")
    scene_name = os.path.basename(scene_dir)
    glb_path = os.path.join(scene_dir, f"{scene_name}.glb")
    navmesh_path = os.path.join(scene_dir, f"{scene_name}.navmesh")

    # ---- 0. The scene mesh must already exist ----
    if not os.path.exists(glb_path):
        raise FileNotFoundError(
            f"Scene mesh not found: {glb_path}\n"
            f"Convention: the --scene-dir directory name is the scene name, and it must contain a matching .glb."
        )

    # ---- 1. The scene-specific config must already be hand-prepared; error out if missing, never auto-generate ----
    yaml_path = f"scripts/eval/configs/vln_r2r_{scene_name}.yaml"
    eval_cfg_path = f"scripts/eval/configs/habitat_dual_system_{scene_name}_cfg.py"
    missing = [p for p in (yaml_path, eval_cfg_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing scene-specific config; the script won't auto-generate it for you "
            "(the model path / output dir need a human decision).\n"
            "Missing file(s):\n  " + "\n  ".join(missing) + "\n"
            f"See section 4 of this file: copy vln_r2r.yaml -> {yaml_path}, "
            f"copy habitat_dual_system_cfg.py -> {eval_cfg_path}, and fix up the paths inside for this scene."
        )

    # ---- 2. Read the actually-effective agent params and dataset paths from the scene-specific yaml ----
    habitat_config = get_habitat_config(yaml_path)
    agent_cfg = habitat_config.habitat.simulator.agents.main_agent
    scenes_dir = habitat_config.habitat.dataset.scenes_dir
    split = habitat_config.habitat.dataset.split
    data_path = habitat_config.habitat.dataset.data_path.format(split=split)

    # ---- 3. navmesh: bake it on the spot if missing, never copy one from elsewhere ----
    ensure_navmesh(glb_path, navmesh_path, agent_cfg.radius, agent_cfg.height)

    # ---- 4. Generate this one single-task episode and write it to the location declared in the yaml ----
    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(navmesh_path)
    assert pf.is_loaded, f"failed to load navmesh: {navmesh_path}"
    
    p0, p1, path = sample_single_episode(pf, args.min_dist, args.max_dist)

    def to_float_list(vec):
        # habitat_sim returns numpy float32 components; json can't serialize those directly
        return [float(x) for x in vec]

    scene_id_rel = os.path.relpath(glb_path, scenes_dir)  # keep consistent with how VLNDatasetV1 joins the path
    episode = {
        "episode_id": 0,
        "trajectory_id": 0,
        "scene_id": scene_id_rel,
        "start_position": to_float_list(p0),
        "start_rotation": yaw_quat(random.uniform(0, 2 * math.pi)),
        "reference_path": [to_float_list(pt) for pt in path.points],
        "goals": [{"position": to_float_list(p1), "radius": 3.0}],
        "instruction": {"instruction_text": args.instruction},
        "info": {"geodesic_distance": float(path.geodesic_distance)},
    }

    payload = {"episodes": [episode], "instruction_vocab": {"word_list": []}}
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with gzip.open(data_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[episode] wrote 1 single-task episode -> {data_path}")
    print(f"          scene_id={scene_id_rel}, geodesic_distance={path.geodesic_distance:.2f}m")

    if args.skip_eval:
        print("--skip-eval was passed, stopping here without running eval.py.")
        return

    # ---- 5. Run one real closed-loop inference pass; this env var guarantees per-step images are saved regardless of success/failure ----
    #         (whether the video is saved depends on vis_debug/save_video in the config, see section 4)
    env = os.environ.copy()
    env["INTERNNAV_SAVE_FRAMES"] = "1"  # with added patch for annotation image output, this parameter controls if such image is saved in logs
    cmd = [sys.executable, "scripts/eval/eval.py", "--config", eval_cfg_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()