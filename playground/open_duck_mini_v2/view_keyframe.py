"""Simple script to view the home keyframe in MuJoCo viewer."""

import mujoco
import mujoco.viewer
import time

# Load the model
model = mujoco.MjModel.from_xml_path(
    "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
)
data = mujoco.MjData(model)

# Set to home keyframe
keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

print("Home keyframe qpos:")
print(data.qpos)
print("\nHome keyframe ctrl:")
print(data.ctrl)

# Launch viewer
print("\nLaunching viewer... Close the window to exit.")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

