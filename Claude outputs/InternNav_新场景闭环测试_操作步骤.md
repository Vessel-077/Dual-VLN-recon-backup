# InternNav 新场景闭环测试 —— 操作步骤(3-8)

> 本文件是完整指南的精简操作版,只保留可执行步骤,背景原理见上一版《InternNav_新场景闭环测试指南.md》。以下命令在 `~/InternNav` 仓库根目录、`conda activate internnav` 之后执行。
>
> 本版按你的新要求调整:场景资产统一放在 `data/other_scene/mini/<场景名>/`;不拷贝别处的 `.navmesh`,一律从 `.glb` 现场烘焙;新增一个一体化脚本,自动完成"定位场景 → 校验 config/navmesh → 生成单任务 episode → 跑闭环 → 保证产出标注视频/图片"。

---

## 3. 场景资产存放位置

只放 `.glb`,不放(也不用拷贝)`.navmesh` —— navmesh 由第 5 步的脚本现场算:

```bash
mkdir -p data/other_scene/mini/random_scene_1
cp /path/to/your_scene.glb data/other_scene/mini/random_scene_1/random_scene_1.glb
```

约定:目录名就是场景名,目录下必须有同名的 `.glb`(例如 `random_scene_1/random_scene_1.glb`)。第 5 步脚本靠这个约定自动定位文件,不用额外传场景名参数。

---

## 4. 场景专属的 config 模板(人工准备一次,脚本只负责校验)

模型路径、输出目录这些需要人工判断的东西,脚本不会替你猜——没有就直接报错。复制现有模板,文件名按场景名命名:

```bash
cp scripts/eval/configs/vln_r2r.yaml \
   scripts/eval/configs/vln_r2r_random_scene_1.yaml
cp scripts/eval/configs/habitat_dual_system_cfg.py \
   scripts/eval/configs/habitat_dual_system_random_scene_1_cfg.py
```

编辑 `vln_r2r_random_scene_1.yaml`,只改 `dataset` 这一段:

```yaml
  dataset:
    type: R2RVLN-v1
    split: single_task
    scenes_dir: data/other_scene/mini
    data_path: data/other_scene/mini/random_scene_1/episodes/{split}.json.gz
```

编辑 `habitat_dual_system_random_scene_1_cfg.py`,注意 `vis_debug` 改成 `True`(原因见前面的结论——这是唯一保证"不管成不成功都有视频"的开关):

```python
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            "model_path": "checkpoints/InternVLA-N1-DualVLN",
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
            "vis_debug": True,   # set to True: per-step annotated video is saved regardless of success or failure
            "vis_debug_path": "./logs/habitat/test_random_scene_1/vis_debug",
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r_random_scene_1.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/test_random_scene_1",
        "save_video": True,           # on success, also saves an extra video annotated with the top-down map
        "epoch": 0,
        "max_steps_per_episode": 200, # a single-task smoke test doesn't need as many as 500
        "port": "2333",
        "dist_url": "env://",
    },
```

`scenes_dir` 是场景目录的父目录(`data/other_scene/mini`),所以 episode 里的 `scene_id` 会写成 `random_scene_1/random_scene_1.glb`;`data_path` 里的 `{split}` 会被自动替换成 `single_task`,其余部分(包括场景名)必须写死在这份场景专属 yaml 里 —— `habitat` 的 dataset loader 只认 `{split}` 这一个占位符,不支持按场景名模板化,这也是为什么每个新场景都要有自己独立的一份 yaml。

---

## 5. 一体化脚本:`run_new_scene_closed_loop.py`

放在 `~/InternNav` 仓库根目录下。逻辑严格按你的要求:解析场景位置 → 校验 config(缺了报错)→ 校验/烘焙 navmesh → 生成 1 条单任务 episode → 跑一次真正的闭环推理。

```python
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
    ap.add_argument("--instruction", default="Walk forward through the space and stop near the far side.")
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
    env["INTERNNAV_SAVE_FRAMES"] = "1"  # requires the section 6 patch first; otherwise this var is never read
    cmd = [sys.executable, "scripts/eval/eval.py", "--config", eval_cfg_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
```

运行示例:

```bash
python run_new_scene_closed_loop.py --scene-dir data/other_scene/mini/random_scene_1
```

---

## 6. 一次性小补丁:让逐帧标注图片不依赖任务是否成功

现有代码里视频写入靠 `imageio` 的 `vis_writer`(第 5 步脚本设的 `INTERNNAV_SAVE_FRAMES` 环境变量目前还没人读取),没有任何地方把标注帧单独存成图片。打开 `internnav/habitat_extensions/vln/habitat_vln_evaluator.py`,搜索 `vis_writer.append_data(vis)`,一共有 **2 处**(分别在 `_run_eval_dual_system` 和 `_run_eval_system2` 两条推理路径里,结构完全一样),在这两处各自后面加 4 行:

```python
                    vis_writer.append_data(vis)
                    # ---- ADDED: also dump this annotated frame as a standalone PNG ----
                    if os.environ.get("INTERNNAV_SAVE_FRAMES"):
                        frames_dir = os.path.join(debug_dir, f'{scene_id}_{episode_id:04d}_frames')
                        os.makedirs(frames_dir, exist_ok=True)
                        Image.fromarray(vis).save(os.path.join(frames_dir, f'step_{step_id:04d}.png'))
```

`Image`(PIL)在文件顶部已经 import 过,不用再加。这段完全由环境变量开关控制,默认不设置就是原来的行为,不会影响你现有跑 MP3D 大规模闭环评测的产出(不会平白多出几万张图片)。

---

## 7. 运行

```bash
cd ~/InternNav
conda activate internnav
python run_new_scene_closed_loop.py --scene-dir data/other_scene/mini/random_scene_1
```

想先只验证场景/episode 是否接入正确、暂时不跑模型(省时间):

```bash
python run_new_scene_closed_loop.py --scene-dir data/other_scene/mini/random_scene_1 --skip-eval
```

---

## 8. 验证输出

- `logs/habitat/test_random_scene_1/progress.json`:这条单任务 episode 的结果(`success`/`spl`/`os`/`ne`/`steps`)。
- `logs/habitat/test_random_scene_1/vis_debug/epoch_0/random_scene_1_0000.mp4`:**逐帧标注视频,不管这次任务成功还是失败,都会有**(因为第 4 步把 `vis_debug` 设成了 `True`)。
- `logs/habitat/test_random_scene_1/vis_debug/epoch_0/random_scene_1_0000_frames/step_XXXX.png`:逐帧标注图片,同样不依赖成功与否(前提是打了第 6 节的补丁,并且第 5 步脚本设置了 `INTERNNAV_SAVE_FRAMES`)。
- `logs/habitat/test_random_scene_1/vis_0/random_scene_1/0000.mp4`:带俯视图/目标点标注的那份"更好看"的视频 —— **只有 `success==1.0` 才会出现**,任务失败时这个文件不会生成,属于正常现象,不代表流程出错。

关于"单任务能否跑通闭环测试并产出标注视频/图片"这个问题的结论已经在文件开头说明:能跑通(`world_size==1` 是代码显式支持的路径),但默认配置下"视频是否落盘"和"任务是否成功"是绑在一起的,本操作文件通过打开 `vis_debug` + 小补丁,把"是否产出标注视频/图片"和"任务是否成功"这两件事解耦开了。
