# 把新场景接入 InternNav / DualVLN 项目并跑通闭环测试

> 基于对你项目文件夹（`InternNav` + `habitat-lab`）的实际代码结构分析编写。这是 [InternRobotics/InternNav](https://github.com/InternRobotics/InternNav) 工具箱，DualVLN 是其中 `InternVLA-N1 (Dual System)` 的一个变体（论文 arXiv:2512.08186）。所有路径、字段名都对照你项目里的真实代码核对过，不是通用 VLN-CE 教程的泛泛之谈。
>
> 下面的命令请在你**跑实验的那台 Linux 机器**上执行（从 `keyboard_nav_server.py` 的注释和 `data/scene_datasets` 的符号链接看，是 `~/InternNav` 仓库根目录，`conda activate internnav` 之后操作）。我看到的 Windows 文件夹只是本地只读副本，不能跑。

---

## 1. 为什么选 habitat_test_scenes 作为"新场景"样例

你要测的是"新建的3D世界模型能否顺利接入项目做闭环实验"这条流程本身，而不是具体某个场景的语义内容，所以样例数据要满足：不需要额外签署许可协议、体积小便于快速迭代、和你现有 MP3D 场景在格式上是同一套 habitat-sim 资产（`.glb` 网格 + `.navmesh` 导航网格），这样才能验证"流程"而不是"格式转换"。

好消息是,**你的项目里已经下载好了这份数据**,我在 `data/versioned_data/habitat_test_scenes_1.0/` 下看到:

```
apartment_1.glb        apartment_1.navmesh
skokloster-castle.glb  skokloster-castle.navmesh
van-gogh-room.glb      van-gogh-room.navmesh
```

并且 `data/scene_datasets/habitat-test-scenes` 是指向这个目录的符号链接,`test_render.py` 已经用它验证过 GPU 渲染是通的。这是 habitat-sim 官方自带的 3 个免许可测试场景(`python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes`下载得到),专门设计来做这种"流程冒烟测试"。

下面以 **`skokloster-castle`**(一座城堡,房间多、路径长,比单个房间的 van-gogh-room 更接近真实 VLN 任务的尺度)为例。如果你想要更快的最小验证,把下面所有 `skokloster-castle` 替换成 `van-gogh-room` 即可,步骤完全一样。

---

## 2. 确认你现有闭环管线的接入方式

先说清楚原理,这样出问题时你知道该查哪里:

- 场景注册在 `scripts/eval/configs/vln_r2r.yaml` 里:`dataset.scenes_dir: data/scene_data/mp3d_ce`,episode 里的 `scene_id` 字段(例如 `mp3d/7y3sRwLe3Va/7y3sRwLe3Va.glb`)会被 habitat 的 `VLNDatasetV1`(`habitat-lab/habitat-lab/habitat/datasets/vln/r2r_vln_dataset.py`)拼接到 `scenes_dir` 后面去定位文件。
- 这份 yaml 里 `simulator.agents.main_agent.sim_sensors` 只配了 `rgb_sensor` 和 `depth_sensor`,**没有语义传感器**,`internnav/habitat_extensions/vln/habitat_vln_evaluator.py` 里我搜了一遍也完全不涉及 `.house` / semantic / scene_dataset_config —— 也就是说,你**不需要**给新场景准备 `.house` 或 `_semantic.ply`,也**不需要**写自定义的 `scene_dataset_config.json`。habitat-sim 会自动识别和 `.glb` 同名的 `.navmesh`。这比通用 Habitat 教程里讲的注册流程简单得多,是专门针对你这套闭环配置的结论。
- Episode 数据集(`data/vln_ce/raw_data/r2r/{split}/{split}.json.gz`)由 `VLNDatasetV1.from_json` 解析,我直接读取了一条真实 episode 确认了字段结构(见下一节)。
- 真正跑闭环推理的入口是 `python scripts/eval/eval.py --config <cfg.py>`,`eval.py` 会动态 import 这个 config 文件里的 `eval_cfg` 对象,交给 `internnav/env/habitat_env.py` 的 `HabitatEnv` 包一层 `habitat.Env`,按 scene 分组、按 rank 分片后逐 episode reset/step。

---

## 3. 第一步:把场景资产放进项目已有的目录约定里

不改任何现有配置,只是照搬你已有 32 个 MP3D 场景的目录结构,新建一个"假装是 mp3d 场景"的文件夹:

```bash
cd ~/InternNav   # 你的 InternNav 仓库根目录
mkdir -p data/scene_data/mp3d_ce/mp3d/skokloster-castle
cp data/versioned_data/habitat_test_scenes_1.0/skokloster-castle.glb \
   data/scene_data/mp3d_ce/mp3d/skokloster-castle/skokloster-castle.glb
cp data/versioned_data/habitat_test_scenes_1.0/skokloster-castle.navmesh \
   data/scene_data/mp3d_ce/mp3d/skokloster-castle/skokloster-castle.navmesh
```

这样它的 `scene_id` 在 episode 里就应该写成 `mp3d/skokloster-castle/skokloster-castle.glb`,和现有 32 个场景的写法(`mp3d/<scene>/​<scene>.glb`)完全一致,`scenes_dir` 不用改。

---

## 4. 第二步:为新场景生成最小闭环测试用的 episode

真实 R2R 数据里的一条 episode 长这样(我从 `val_unseen.json.gz` 里原样读出来的):

```json
{
  "episode_id": 1,
  "trajectory_id": 4,
  "scene_id": "mp3d/7y3sRwLe3Va/7y3sRwLe3Va.glb",
  "start_position": [-16.267, 0.152, 0.721],
  "start_rotation": [-0.0, 0.707, -0.0, -0.707],
  "reference_path": [[...], [...], ...],
  "goals": [{"position": [-12.337, 0.152, 4.214], "radius": 3.0}],
  "instruction": {"instruction_text": "...", "instruction_tokens": [...]},
  "info": {"geodesic_distance": 6.43}
}
```

另外我确认了两个容易踩坑的地方(直接读了 `VLNDatasetV1.from_json` 源码):

1. json 顶层除了 `"episodes"` 之外,**必须**有 `"instruction_vocab": {"word_list": [...]}` 这个 key,否则会 `KeyError`(`word_list` 可以是空列表,`VocabDict` 会自动补 `<unk>`)。
2. `instruction_tokens` 完全没有被 `habitat_vln_evaluator.py` 用到(它只读 `episode.instruction.instruction_text`),所以可以留空或不写。

由于是自建场景,没有真实人工标注的指令和轨迹,我们只需要**几何上合法**的 start/goal 对(在 navmesh 上可达、有有限测地距离),指令文字随便写一句描述性的话就行 —— 目的是验证管线跑通,不是评测语言理解质量。

在 `~/InternNav` 下新建 `make_custom_scene_episodes.py`:

```python
#!/usr/bin/env python
"""为新场景(默认 skokloster-castle)生成一个最小闭环测试用的 R2R 格式 episode 文件。"""
import argparse
import gzip
import json
import math
import random

import habitat_sim


def yaw_quat(angle_rad):
    return [0.0, math.sin(angle_rad / 2), 0.0, math.cos(angle_rad / 2)]


def sample_episode_pair(pathfinder, min_dist=3.0, max_dist=12.0, max_tries=200):
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
    raise RuntimeError("采样多次仍未找到满足距离要求的合法起止点，试着放宽 min_dist/max_dist")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-id", default="skokloster-castle",
                     help="对应 data/scene_data/mp3d_ce/mp3d/<scene-id>/ 下的文件夹名")
    ap.add_argument("--navmesh", default=None,
                     help="默认 data/scene_data/mp3d_ce/mp3d/<scene-id>/<scene-id>.navmesh")
    ap.add_argument("--out", default=None,
                     help="默认 data/vln_ce/raw_data/r2r_custom_scene/val_mini/val_mini.json.gz")
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--instruction", default="Walk forward through the room and stop near the far wall.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    navmesh = args.navmesh or f"data/scene_data/mp3d_ce/mp3d/{args.scene_id}/{args.scene_id}.navmesh"
    out_path = args.out or "data/vln_ce/raw_data/r2r_custom_scene/val_mini/val_mini.json.gz"

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(navmesh)
    assert pf.is_loaded, f"navmesh 加载失败: {navmesh}"

    episodes = []
    for i in range(args.num_episodes):
        p0, p1, path = sample_episode_pair(pf)
        episodes.append({
            "episode_id": i,
            "trajectory_id": i,
            "scene_id": f"mp3d/{args.scene_id}/{args.scene_id}.glb",
            "start_position": list(p0),
            "start_rotation": yaw_quat(random.uniform(0, 2 * math.pi)),
            "reference_path": [list(pt) for pt in path.points],
            "goals": [{"position": list(p1), "radius": 3.0}],
            "instruction": {"instruction_text": args.instruction},
            "info": {"geodesic_distance": path.geodesic_distance},
        })

    payload = {"episodes": episodes, "instruction_vocab": {"word_list": []}}

    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"写入 {len(episodes)} 条 episode -> {out_path}")
    for ep in episodes:
        print(f"  ep{ep['episode_id']}: geodesic_distance={ep['info']['geodesic_distance']:.2f}m, "
              f"start={ep['start_position']}, goal={ep['goals'][0]['position']}")


if __name__ == "__main__":
    main()
```

运行(只需要 `habitat_sim`,不需要 GPU/显示,`PathFinder` 是纯 CPU 的导航网格计算,不会跟你后面跑模型抢显存):

```bash
python make_custom_scene_episodes.py --scene-id skokloster-castle --num-episodes 3
```

会在 `data/vln_ce/raw_data/r2r_custom_scene/val_mini/val_mini.json.gz` 生成一个全新的、独立于你现有 R2R 数据集的小 split —— 完全不会碰到你已经跑通的 MP3D 数据。

想直观看一眼采样是否合理,可以照搬你已有的 `browse_vlnce_dataset.py` 套路:

```bash
python browse_vlnce_dataset.py --data-root data/vln_ce/raw_data/r2r_custom_scene --split val_mini
```

会在 `vlnce_overview/val_mini/` 下产出 CSV + 参考路径俯视图,跟你现有习惯的检查方式一致。

---

## 5. 第三步:新建一份指向新场景的 habitat yaml + eval config

复制现有的两个文件,只改必要字段,**不动原文件**,这样你随时能切回原来的 MP3D 闭环实验:

```bash
cp scripts/eval/configs/vln_r2r.yaml scripts/eval/configs/vln_r2r_custom_scene.yaml
cp scripts/eval/configs/habitat_dual_system_cfg.py scripts/eval/configs/habitat_dual_system_custom_cfg.py
```

编辑 `vln_r2r_custom_scene.yaml`,只改 `dataset` 这一段(其余 sensor/action/measurement 配置照搬即可,新场景一样用得上):

```yaml
  dataset:
    type: R2RVLN-v1
    split: val_mini
    scenes_dir: data/scene_data/mp3d_ce
    data_path: data/vln_ce/raw_data/r2r_custom_scene/{split}/{split}.json.gz
```

编辑 `habitat_dual_system_custom_cfg.py`,只改这两处(agent/checkpoint 保持不变,复用你已经验证过的 `InternVLA-N1-DualVLN`):

```python
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r_custom_scene.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/test_custom_scene",
        "save_video": True,          # 建议开着，第一次跑肉眼确认在新场景里真的动了
        "epoch": 0,
        "max_steps_per_episode": 200, # 冒烟测试不用给 500 那么多
        "port": "2333",
        "dist_url": "env://",
    },
```

---

## 6. (强烈建议)先用纯 habitat.Env 冒烟测试一遍数据管线,不跑模型

新场景接入最容易出问题的是"数据格式对不对",而不是"模型好不好",先隔离这一层能省很多调试时间。写一个 `smoke_test_custom_scene.py`:

```python
import habitat

config = habitat.get_config("scripts/eval/configs/vln_r2r_custom_scene.yaml")
with habitat.Env(config=config) as env:
    print("加载到的场景数:", len(env.episodes))
    obs = env.reset()
    print("观测 keys:", list(obs.keys()), "rgb shape:", obs["rgb"].shape)
    for _ in range(10):
        obs = env.step("move_forward")
    print("走了 10 步之后的 metrics:", env.get_metrics())
```

```bash
python smoke_test_custom_scene.py
```

能正常打印、不报错,说明场景注册、episode 格式、navmesh 都没问题 —— 这一步验证的正是你想测的"新场景能否顺利接入项目"这件事本身。

---

## 7. 第四步:正式跑闭环推理

确认第 6 步没问题之后,再跑真正会调用 `InternVLA-N1-DualVLN` 做逐步推理的完整闭环评测:

```bash
python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_custom_cfg.py
```

单机单卡就是这一条命令(和你现在跑 MP3D 闭环用的是同一个 `eval.py`,只是换了 `--config`)。如果你现在用的是多卡/SLURM,对应换成:

```bash
./scripts/eval/bash/torchrun_eval.sh --config scripts/eval/configs/habitat_dual_system_custom_cfg.py
```

---

## 8. 验证是否真的跑通了

- `logs/habitat/test_custom_scene/progress.json`:逐 episode 的结果(resumability 用的那份),每一行应该出现 `"scene_id": "skokloster-castle"` 或类似字段,以及 `success` / `spl` / `oracle_success` / `oracle_navigation_error` 这几个 measurement。
- `logs/habitat/test_custom_scene/result.json`:所有 episode 跑完后的聚合指标。
- 因为 `save_video: True`,`vis_debug_path` 也开着,可以直接把生成的视频拉出来看一眼,agent 是不是真的在这座城堡里走动、有没有一直撞墙/原地转圈(几何上不合理的话,大概率是采样出的 start/goal 距离太远/隔着房间墙,回第 4 步调大 `--num-episodes` 重新采样,或者调小 `max_dist`)。

跑通以上两点,就证明"新场景接入项目做闭环实验"这条链路是打通的:场景注册、episode 生成、eval config、闭环推理、指标落盘,全流程都验证过了。

---

## 9. 之后如果要换成真正"自己搭建"的3D世界(不只是官方样例)

流程完全一样,唯一的区别是第 1 步换成:

1. 用 Blender / 扫描重建工具导出 `.glb`(注意坐标系:habitat 默认 Y-up,如果你的建模工具是 Z-up 需要在导出时转换,否则导航网格会歪掉)。
2. 用 habitat-sim 重新烘焙导航网格(而不是直接复用别人的 `.navmesh`):
   ```python
   import habitat_sim
   sim_cfg = habitat_sim.SimulatorConfiguration()
   sim_cfg.scene_id = "path/to/your_scene.glb"
   sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [habitat_sim.agent.AgentConfiguration()]))
   sim.recompute_navmesh(sim.pathfinder, habitat_sim.NavMeshSettings())
   sim.pathfinder.save_nav_mesh("path/to/your_scene.navmesh")
   ```
3. 从这里开始,回到本指南第 3 步,把 `.glb` + `.navmesh` 放进 `data/scene_data/mp3d_ce/mp3d/<你的场景名>/`,后面第 4-8 步一字不改就能复用。

---

### 涉及到的文件一览(便于回头核对)

| 文件 | 作用 |
|---|---|
| `data/scene_data/mp3d_ce/mp3d/<scene>/<scene>.glb` + `.navmesh` | 场景资产,新场景要放在这 |
| `data/vln_ce/raw_data/r2r_custom_scene/val_mini/val_mini.json.gz` | 新生成的最小 episode 数据集 |
| `scripts/eval/configs/vln_r2r_custom_scene.yaml` | habitat 侧配置(sensor/action/measurement/dataset 路径) |
| `scripts/eval/configs/habitat_dual_system_custom_cfg.py` | InternNav 侧 eval 入口配置(agent/env/eval_settings) |
| `scripts/eval/eval.py` | 闭环评测主入口 |
| `internnav/env/habitat_env.py` | 把 habitat.Env 包装成 InternNav 的 Env 接口 |
| `internnav/habitat_extensions/vln/habitat_vln_evaluator.py` | 真正跑模型推理 + 计算指标的闭环循环 |
| `habitat-lab/habitat-lab/habitat/datasets/vln/r2r_vln_dataset.py` | episode json 的解析逻辑(字段要求的权威来源) |
