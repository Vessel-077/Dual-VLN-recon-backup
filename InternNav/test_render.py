import habitat_sim

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

sensor_cfg = habitat_sim.CameraSensorSpec()
sensor_cfg.uuid = "color_sensor"
sensor_cfg.resolution = [256, 256]

agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sensor_cfg]

sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
obs = sim.reset()
print("渲染成功,画面尺寸:", obs["color_sensor"].shape)
sim.close()
print("GPU rendering OK")
