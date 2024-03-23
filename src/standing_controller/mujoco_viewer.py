from dm_control import mujoco
from mujoco import viewer
import os
import pinocchio as pin

from instant_qp_wbc import InstantQPWBC
from mj_pin_util import *

os.environ["MUJOCO_GL"] = "egl"
BASE_DIR = "/home/keseterg/Documents/Learning/robotics/control_demos"

physics = mujoco.Physics.from_xml_path(
    os.path.join(BASE_DIR, "models", "digit", "digit-v3.xml")
)
robot = pin.RobotWrapper.BuildFromURDF(
    os.path.join(BASE_DIR, "models", "digit", "urdf", "digit_float.urdf"),
    package_dirs=[
        os.path.join(BASE_DIR, "models", "digit", "urdf")
    ],
    root_joint=pin.JointModelFreeFlyer()
)

qb_wbc = InstantQPWBC(robot.model, robot.data)
qb_wbc.initialize()


def update_sim():
    physics.step()
    physics.forward()


def update_wbc_control():
    qb_wbc.update_states(
        mj_to_pin_q(physics.data.qpos),
        mj_to_pin_qd(physics.data.qvel),
        mj_to_pin_qd(physics.data.qacc_smooth),
    )
    computed_tau = qb_wbc.solve_torque(verbose=True)

    physics.data.ctrl = pin_to_mj_tau(computed_tau)



with viewer.launch_passive(physics.model._model, physics.data._data) as v:
    # update functions
    while v.is_running():
        update_sim()
        update_wbc_control()
        v.sync()
