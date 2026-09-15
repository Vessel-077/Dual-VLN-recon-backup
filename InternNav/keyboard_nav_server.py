#!/usr/bin/env python
"""
Keyboard-controlled Habitat navigation, streamed to a browser via MJPEG.

Run on the remote workstation (inside the `internnav` conda env, from the
InternNav repo root), then open the forwarded port in your LOCAL browser
(VS Code Remote-SSH auto-forwards it) to drive the agent with WASD / arrow
keys and export what you did as a pure-camera .mp4.

IMPORTANT: habitat-sim's OpenGL/EGL context is bound to whichever thread
created the Simulator. Flask's threaded=True mode handles each request on
a new thread, so simulator calls (step/reset/render) must never be made
directly from a Flask request handler -- they must all be dispatched to
one dedicated "sim thread" via a job queue. That's what submit_job() /
sim_worker() below do.
"""
import argparse
import io
import os
import queue
import threading
import time
from concurrent.futures import Future

import numpy as np
import habitat_sim
import imageio.v2 as imageio
from flask import Flask, Response, request, jsonify
from PIL import Image

lock = threading.Lock()
state = {"frame": None, "recording": [], "step_count": 0}

job_queue = queue.Queue()
ready_event = threading.Event()

app = Flask(__name__)

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Habitat Keyboard Nav</title>
<style>
  body { background:#111; color:#eee; font-family: sans-serif; text-align:center; }
  img { border: 2px solid #444; margin-top: 20px; }
  .hint { margin-top: 10px; font-size: 14px; color:#aaa; }
  button { margin: 10px; padding: 8px 16px; font-size: 14px; }
  #status { margin-top: 10px; font-size: 14px; color: #6f6; }
</style>
</head>
<body>
  <h2>Habitat Keyboard Navigation (click the view first to focus it)</h2>
  <img id="cam" src="/stream" width="640" height="480" tabindex="0">
  <div class="hint">W / &uarr; Forward&nbsp;&nbsp; A / &larr; Turn left&nbsp;&nbsp; D / &rarr; Turn right&nbsp;&nbsp; R Reset position</div>
  <div>
    <button onclick="doReset()">Reset position</button>
    <button onclick="saveVideo()">Export this run as video</button>
  </div>
  <div id="status">Steps: 0</div>
<script>
let busy = false;
async function sendAction(action) {
  if (busy) return;
  busy = true;
  try {
    const res = await fetch("/action", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action})
    });
    const data = await res.json();
    if (data.ok) document.getElementById("status").innerText = "Steps: " + data.step;
    else document.getElementById("status").innerText = "Error: " + data.error;
  } finally { busy = false; }
}
async function doReset() {
  const res = await fetch("/reset", {method: "POST"});
  const data = await res.json();
  document.getElementById("status").innerText = data.ok ? "Steps: 0 (reset)" : ("Error: " + data.error);
}
async function saveVideo() {
  document.getElementById("status").innerText = "Exporting video...";
  const res = await fetch("/save_video", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({fps: 6})
  });
  const data = await res.json();
  document.getElementById("status").innerText = data.ok
    ? ("Video saved: " + data.path + " (" + data.num_frames + " frames)")
    : ("Export failed: " + data.error);
}
document.addEventListener("keydown", (e) => {
  const key = e.key.toLowerCase();
  if (key === "w" || key === "arrowup") sendAction("move_forward");
  else if (key === "a" || key === "arrowleft") sendAction("turn_left");
  else if (key === "d" || key === "arrowright") sendAction("turn_right");
  else if (key === "r") doReset();
});
</script>
</body>
</html>
"""


def get_rgb_from_obs(obs):
    return np.asarray(obs["color_sensor"])[:, :, :3]


def encode_jpeg(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def make_sim(scene_glb, width, height, gpu_device_id):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    sim_cfg.gpu_device_id = gpu_device_id

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.resolution = [height, width]
    sensor_spec.position = [0.0, 1.25, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def place_random(sim):
    pt = sim.pathfinder.get_random_navigable_point()
    agent = sim.get_agent(0)
    agent_state = agent.get_state()
    agent_state.position = pt
    agent.set_state(agent_state)


def submit_job(fn, timeout=30):
    """Run fn(sim) on the dedicated sim thread and block for its result.
    Call this from Flask request handlers instead of touching the
    simulator directly -- the GL context only lives on the sim thread."""
    fut = Future()
    job_queue.put((fn, fut))
    return fut.result(timeout=timeout)


def sim_worker(scene_glb, width, height, gpu_device_id):
    """Owns the Simulator (and its GL context) for the whole process
    lifetime. Everything that touches `sim` happens here, sequentially."""
    sim = make_sim(scene_glb, width, height, gpu_device_id)
    place_random(sim)
    obs = sim.get_sensor_observations()
    with lock:
        state["frame"] = get_rgb_from_obs(obs)
    ready_event.set()

    while True:
        fn, fut = job_queue.get()
        try:
            result = fn(sim)
            fut.set_result(result)
        except Exception as e:  # noqa: BLE001
            fut.set_exception(e)


@app.route("/")
def index():
    return PAGE


@app.route("/stream")
def stream():
    def gen():
        while True:
            with lock:
                frame = state["frame"]
            if frame is not None:
                jpg = encode_jpeg(frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/action", methods=["POST"])
def action():
    act = (request.json or {}).get("action")
    if act not in ("move_forward", "turn_left", "turn_right"):
        return jsonify({"ok": False, "error": "bad action"}), 400

    def job(sim):
        obs = sim.step(act)
        frame = get_rgb_from_obs(obs)
        with lock:
            state["frame"] = frame
            state["recording"].append(frame.copy())
            state["step_count"] += 1
            n = state["step_count"]
        return n

    try:
        n = submit_job(job)
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, "step": n})


@app.route("/reset", methods=["POST"])
def reset():
    def job(sim):
        place_random(sim)
        obs = sim.get_sensor_observations()
        frame = get_rgb_from_obs(obs)
        with lock:
            state["frame"] = frame
            state["recording"] = []
            state["step_count"] = 0
        return True

    try:
        submit_job(job)
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True})


@app.route("/save_video", methods=["POST"])
def save_video():
    fps = int((request.json or {}).get("fps", 6))
    out_dir = os.environ.get("NAV_VIDEO_DIR", "./keyboard_nav_out")
    os.makedirs(out_dir, exist_ok=True)
    with lock:
        frames = list(state["recording"])
    if not frames:
        return jsonify({"ok": False, "error": "no actions recorded yet"}), 400
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"keyboard_nav_{ts}.mp4")
    writer = imageio.get_writer(path, fps=fps, macro_block_size=None)
    for f in frames:
        writer.append_data(f)
    writer.close()
    return jsonify({"ok": True, "path": os.path.abspath(path), "num_frames": len(frames)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="scene id, e.g. zsNo4HB9uLZ")
    p.add_argument("--scenes-root", default="data/scene_data/mp3d_ce/mp3d")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    scene_glb = os.path.join(args.scenes_root, args.scene, f"{args.scene}.glb")
    if not os.path.exists(scene_glb):
        raise FileNotFoundError(f"scene not found: {scene_glb}")

    print(f"[1/2] Starting sim thread for {scene_glb} ...", flush=True)
    worker = threading.Thread(
        target=sim_worker,
        args=(scene_glb, args.width, args.height, args.gpu_id),
        daemon=True,
    )
    worker.start()
    ready_event.wait()
    print("[2/2] Simulator ready. Starting Flask server ...", flush=True)

    print(f"Scene loaded: {scene_glb}")
    print(f"Open http://localhost:{args.port} in your LOCAL browser (via the forwarded port).")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()