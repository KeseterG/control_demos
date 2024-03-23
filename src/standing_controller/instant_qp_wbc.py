import time

import numpy as np
import pinocchio as pin
from pinocchio import ReferenceFrame

from common_types import *
from qp_formulator import *


class InstantQPWBC(object):
    def __init__(self, model: PModel, data: PData) -> None:
        self.model: PModel = model
        self.data: PData = data

        # ------- Constants -------
        # Motion
        self.nq = self.model.nq  # joint angles dim
        self.nv = self.model.nv  # joint vel / acceleration dim
        self.k_passive_id = [0, 1, 2, 3, 4, 5, 10, 17]
        self.k_act_id = [6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]  # rod, 9=10, 16=17
        self.nact = len(self.k_act_id)
        self.S = np.zeros((self.nv, self.nact))
        counter = 0
        for row in self.k_act_id:
            self.S[row][counter] = 1
            counter += 1

        self.k_tau_abs_max = 30.0
        self.k_tau_abs_limit = np.zeros(self.nv)
        for act in self.k_act_id:
            self.k_tau_abs_limit[act] = self.k_tau_abs_max

        # Contact
        self.n_contact = 2  # double support phase, act on toe pitch joint
        self.nd = 5  # polyhedral approximation of the friction cone
        self.k_contact_jnt_names = ["toe_pitch_joint_left", "toe_pitch_joint_right"]
        self.k_contact_jnt_ids = [self.model.getJointId(jnt) for jnt in self.k_contact_jnt_names]
        self.k_contact_frame_names = ["toe_pitch_joint_left", "toe_pitch_joint_right"]
        self.k_contact_frame_ids = [self.model.getFrameId(frame) for frame in self.k_contact_frame_names]
        self.k_mu = 1.0  # friction coefficient

        # Dimensions
        self.ndim_beta = self.n_contact * self.nd
        self.ndim_slack = 6 * self.n_contact + 1  # slack for slipping and contact, [eta alpha]
        # decision variables: [ qdd tau beta eta alpha ]
        self.ndim_decision = self.nv + self.nact + self.ndim_beta + self.ndim_slack
        self.dim_s = [0, self.nv, self.nv + self.nact, self.nv + self.nact + self.ndim_beta, self.ndim_decision]

        # Parameters
        self.k_omega_q = 0.1  # tracking cost
        self.k_epsilon = 1e-7  # force polyhedral param
        self.k_eta_min = -5.0  # slack var min
        self.k_eta_max = 5.0  # slack var max
        self.k_alpha_min = -10.0
        self.k_alpha_max = 10.0

        # ------- QP Formulation -------
        # QP decision variables: [ qdd beta slack_vars ].T
        self.qp = QPFormulator(self.ndim_decision)
        self.task_updates: List[Callable] = []

        # ------- State Variables -------
        self.q = np.zeros(self.nq)
        self.qd = np.zeros(self.nv)
        self.qdd = np.zeros(self.nv)
        self.qdd_des = np.zeros(self.nv)
        self.solve_count = 0

    def initialize(self):
        self.__compute_system_status()
        self.task_updates.extend(
            [
                self.__update_tracking_task,
                self.__update_optimal_control_task,
                self.__update_optimal_contact_task,
                # self.__update_slack_var_cost_task,

                self.__update_dynamics_constraint,
                self.__update_rod_constraint,
                # self.__update_contact_no_slip_constraint,
                self.__update_decision_var_bound_constraint
            ]
        )

        for formulate_task in self.task_updates:
            self.qp.add_task(formulate_task())
            self.qp.tasks[-1].check()

    def update_states(self, q, qd, qdd):
        self.q = q
        self.qd = qd
        self.qdd = qdd

    def solve_torque(self, verbose=False) -> Vector:
        self.__compute_system_status()
        self.__update_tasks()

        start_time = time.time()
        solution = self.qp.solve()
        if verbose or not solution.found:
            print(f"QP Problem {self.solve_count} {'successfully solved.' if solution.found else 'FAILED.'} ")
            print(f"- Primal residual: {solution.primal_residual():.1e}")
            print(f"- Dual residual: {solution.dual_residual():.1e}")
            print(f"- Duality gap: {solution.duality_gap():.1e}")
            print(f"- Solve time: {(time.time() - start_time) * 1000:.4f}ms")
        self.solve_count += 1
        return solution.x[self.dim_s[1]:self.dim_s[2]] if solution.found else np.zeros(self.nact)

    def __compute_system_status(self) -> None:
        # kinematics updates
        pin.forwardKinematics(self.model, self.data, self.q, self.qd)

        # jacobians
        pin.computeJointJacobians(self.model, self.data, self.q)
        pin.computeJointJacobiansTimeVariation(self.model, self.data, self.q, self.qd)

        # dynamics
        pin.crba(self.model, self.data, self.q)
        pin.nonLinearEffects(self.model, self.data, self.q, self.qd)

    def __update_tasks(self):
        for i in range(len(self.qp.tasks)):
            self.qp.tasks[i] = self.task_updates[i]()

    def __update_tracking_task(self) -> Task:
        P = np.zeros((self.ndim_decision, self.ndim_decision))
        P[self.dim_s[0]:self.dim_s[1], self.dim_s[0]:self.dim_s[1]] = np.eye(self.nv)
        q = np.zeros(self.ndim_decision)
        q[0:self.nv] = -self.qdd_des

        return Task(TaskType.COST, P=P, q=q)

    def __update_optimal_control_task(self) -> Task:
        P = np.zeros((self.ndim_decision, self.ndim_decision))
        P[self.dim_s[1]:self.dim_s[2], self.dim_s[1]:self.dim_s[2]] = np.eye(self.nact)
        q = np.zeros(self.ndim_decision)

        return Task(TaskType.COST, P=P, q=q)

    def __update_optimal_contact_task(self) -> Task:
        P = np.zeros((self.ndim_decision, self.ndim_decision))
        P[self.dim_s[2]:self.dim_s[3], self.dim_s[2]:self.dim_s[3]] = np.eye(self.ndim_beta) * self.k_omega_q
        q = np.zeros(self.ndim_decision)

        return Task(TaskType.COST, P=P, q=q)

    def __update_slack_var_cost_task(self) -> Task:
        P = np.zeros((self.ndim_decision, self.ndim_decision))
        P[self.dim_s[3]:self.dim_s[4] - 1, self.dim_s[3]:self.dim_s[4] - 1] = np.eye(
            self.ndim_slack - 1
        ) * 2  # eta cost only
        q = np.zeros(self.ndim_decision)

        return Task(TaskType.COST, P=P, q=q)

    def __update_dynamics_constraint(self) -> Task:
        M = self.data.M  # mass matrix, shape: nv * nv
        nle = self.data.nle
        J_c = np.hstack(
            [
                pin.getJointJacobian(self.model, self.data, jnt_id, ReferenceFrame.LOCAL_WORLD_ALIGNED).T
                for jnt_id in self.k_contact_jnt_ids
            ]
        )  # contact jacobian, shape: nv * (6 * n_contact)

        polyhedral = np.array(
            [
                pin.Force(linear=np.array([0, 0, 1]), angular=np.zeros(3)).vector,
                pin.Force(linear=np.array([-self.k_mu, 0, 1]), angular=np.zeros(3)).vector,
                pin.Force(linear=np.array([self.k_mu, 0, 1]), angular=np.zeros(3)).vector,
                pin.Force(linear=np.array([0, -self.k_mu, 1]), angular=np.zeros(3)).vector,
                pin.Force(linear=np.array([0, self.k_mu, 1]), angular=np.zeros(3)).vector,
            ]
        ).T  # polyhedral approximation: [normal vector, 4 polyhedral edge]

        Mat_contact = block_diag(
            *[polyhedral for _ in range(self.n_contact)]
        )  # shape: (6 * n_contact) * n_beta
        Psi = J_c @ Mat_contact  # mapping matrix, beta -(polyhedral)-> forces -(J.T)-> torque, shape: nv * n_beta

        task = Task(TaskType.EQUALITY_CONSTRAINT)
        # M qdd - tau - Phi beta = -nle
        task.A = np.hstack([M, -self.S @ np.eye(self.nact), -Psi, np.zeros((self.nv, self.ndim_slack))])
        task.b = -nle

        return task

    def __update_contact_no_slip_constraint(self) -> Task:
        task = Task(TaskType.EQUALITY_CONSTRAINT)

        J_c = np.vstack([
            pin.getJointJacobian(
                self.model, self.data, jnt_id, ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            for jnt_id in self.k_contact_jnt_ids
        ])
        Jd_c = np.vstack([
            pin.getJointJacobianTimeVariation(
                self.model, self.data, jnt_id, ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            for jnt_id in self.k_contact_jnt_ids
        ])

        task.A = np.zeros((12, self.ndim_decision))
        task.A[0:12, self.dim_s[0]:self.dim_s[1]] = J_c
        task.A[0:12, self.dim_s[3]:self.dim_s[4] - 1] = np.eye(12)
        task.A[0:12, self.dim_s[4] - 1:self.dim_s[4]] = (J_c @ self.qd).reshape((12, 1))
        task.b = -Jd_c @ self.qd

        return task

    def __update_rod_constraint(self) -> Task:
        task = Task(TaskType.EQUALITY_CONSTRAINT)
        task.A = np.zeros((2, self.ndim_decision))
        # accel on the ends of the rod should be equal but opposite
        task.A[0][9] = 1
        task.A[0][10] = 1
        task.A[1][16] = 1
        task.A[1][17] = 1
        task.b = np.zeros(2)

        return task

    def __update_decision_var_bound_constraint(self) -> Task:
        task = Task(TaskType.LB_UB)
        task.lb = np.hstack([
            np.ones(self.nv) * -np.infty,
            -np.ones(self.nact) * self.k_tau_abs_max,
            np.zeros(self.ndim_beta),
            np.ones(self.ndim_slack - 1) * self.k_eta_min,
            np.array([self.k_alpha_min])
        ]).T.reshape((self.ndim_decision,))
        task.ub = np.hstack([
            np.ones(self.nv) * np.infty,
            np.ones(self.nact) * self.k_tau_abs_max,
            np.ones(self.ndim_beta) * np.infty,
            np.ones(self.ndim_slack - 1) * self.k_eta_max,
            np.array([self.k_alpha_max])
        ]).T.reshape((self.ndim_decision,))

        # lock upper mechanism, prevent arm movement
        task.lb[self.dim_s[1] + 12:self.dim_s[2]] = 0
        task.ub[self.dim_s[1] + 12:self.dim_s[2]] = 0

        return task
