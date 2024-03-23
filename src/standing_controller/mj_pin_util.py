import numpy as np

q_delta = 5
qd_delta = 4
nq = 29
nv = 28

"""
 0        left-leg/hip-roll   
 1         left-leg/hip-yaw   
 2       left-leg/hip-pitch   
 3            left-leg/knee   
 4           left-leg/toe-a   
 5           left-leg/toe-b   
 6   left-arm/shoulder-roll   
 7  left-arm/shoulder-pitch   
 8    left-arm/shoulder-yaw   
 9           left-arm/elbow   
10       right-leg/hip-roll   
11        right-leg/hip-yaw   
12      right-leg/hip-pitch   
13           right-leg/knee   
14          right-leg/toe-a   
15          right-leg/toe-b   
16  right-arm/shoulder-roll   
17 right-arm/shoulder-pitch   
18   right-arm/shoulder-yaw   
19          right-arm/elbow  

Nb joints = 24 (nq=29,nv=28)
  Joint 0 universe: parent=0
  Joint 1 root_joint: parent=0
  Joint 2 hip_abduction_left: parent=1
  Joint 3 hip_rotation_left: parent=2
  Joint 4 hip_flexion_left: parent=3
  Joint 5 knee_joint_left: parent=4
  Joint 6 shin_to_tarsus_left: parent=5
  Joint 7 toe_pitch_joint_left: parent=6
  Joint 8 toe_roll_joint_left: parent=7
  Joint 9 hip_abduction_right: parent=1
  Joint 10 hip_rotation_right: parent=9
  Joint 11 hip_flexion_right: parent=10
  Joint 12 knee_joint_right: parent=11
  Joint 13 shin_to_tarsus_right: parent=12
  Joint 14 toe_pitch_joint_right: parent=13
  Joint 15 toe_roll_joint_right: parent=14
  Joint 16 shoulder_roll_joint_left: parent=1
  Joint 17 shoulder_pitch_joint_left: parent=16
  Joint 18 shoulder_yaw_joint_left: parent=17
  Joint 19 elbow_joint_left: parent=18
  Joint 20 shoulder_roll_joint_right: parent=1
  Joint 21 shoulder_pitch_joint_right: parent=20
  Joint 22 shoulder_yaw_joint_right: parent=21
  Joint 23 elbow_joint_right: parent=22 
"""

def mj_to_pin_q(q: np.array) -> np.array:
    result = np.zeros(nq)

    # copy floating joint
    result[0:3] = q[0:3].copy()
    result[3] = q[4]
    result[4] = q[5]
    result[5] = q[6]
    result[6] = q[3]

    # copy the corresponding joints
    result[2 + q_delta] = q[7]
    result[3 + q_delta] = q[8]
    result[4 + q_delta] = q[9]
    result[5 + q_delta] = q[14]
    result[6 + q_delta] = q[15]
    result[7 + q_delta] = q[15]
    result[8 + q_delta] = q[15]
    result[9 + q_delta] = q[34]
    result[10 + q_delta] = q[35]
    result[11 + q_delta] = q[36]
    result[12 + q_delta] = q[41]
    result[13 + q_delta] = q[42]
    result[14 + q_delta] = q[45]
    result[15 + q_delta] = q[50]
    result[16 + q_delta] = q[30]
    result[17 + q_delta] = q[31]
    result[18 + q_delta] = q[32]
    result[19 + q_delta] = q[33]
    result[20 + q_delta] = q[57]
    result[21 + q_delta] = q[58]
    result[22 + q_delta] = q[59]
    result[23 + q_delta] = q[60]

    return result


def mj_to_pin_qd(q: np.array) -> np.array:
    result = np.zeros(nv)

    # copy floating joint
    result[0:6] = q[0:6].copy()

    # copy the corresponding joints
    result[2 + qd_delta] = q[6]
    result[3 + qd_delta] = q[7]
    result[4 + qd_delta] = q[8]
    result[5 + qd_delta] = q[12]
    result[6 + qd_delta] = q[13]
    result[7 + qd_delta] = q[16]
    result[8 + qd_delta] = q[20]
    result[9 + qd_delta] = q[30]
    result[10 + qd_delta] = q[31]
    result[11 + qd_delta] = q[12]
    result[12 + qd_delta] = q[36]
    result[13 + qd_delta] = q[37]
    result[14 + qd_delta] = q[40]
    result[15 + qd_delta] = q[44]
    result[16 + qd_delta] = q[26]
    result[17 + qd_delta] = q[27]
    result[18 + qd_delta] = q[28]
    result[19 + qd_delta] = q[29]
    result[20 + qd_delta] = q[50]
    result[21 + qd_delta] = q[51]
    result[22 + qd_delta] = q[52]
    result[23 + qd_delta] = q[53]

    return result
def pin_to_mj_tau(tau: np.array) -> np.array:
    tau_des = np.zeros_like(tau)
    tau_des[0:6] = tau[0:6].copy()
    tau_des[6:10] = tau[12:16].copy()
    tau_des[10:16] = tau[6:12].copy()
    tau_des[16:20] = tau[16:20].copy()

    return tau_des

