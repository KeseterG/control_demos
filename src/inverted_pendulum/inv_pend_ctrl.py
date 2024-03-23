from dm_control import mujoco
from mujoco import viewer
import os

os.environ["MUJOCO_GL"] = "egl"
NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
BASE_DIR = "/home/keseterg/Documents/Learning/robotics/control_demos"
if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
        f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")


# Utility Function Collection
def get_model_path(pkg_name: str, name: str=None) -> str:
    if not name:
        name = pkg_name
    return os.path.join(
        BASE_DIR, "models",
        pkg_name, f"{name}.xml"
    )

physics = mujoco.Physics.from_xml_path(get_model_path("inverted_pendulum"))


def update_sim():
    physics.step()
    physics.forward()

with viewer.launch_passive(physics.model._model, physics.data._data) as v:
    # update functions
    while v.is_running():
        update_sim()
        v.sync()
