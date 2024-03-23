import numpy as np
from scipy.linalg import block_diag
from typing import Type

import osqp

Vector = np.array


class InvertedPendulum2D:
    """
    Inverted Pendulum in 2D.
    Joint space is defined the angle, velocity and acceleration of the motor.
    Task space is defined as the Cartesian Coordinate of the mass.
    State: [theta, theta_dot]
    Control Input: [tau]
    CCW positive, starting at 0, mass vertically upwards
    """
    g = 9.81
    dt = 0.01

    def __init__(self, R, m) -> None:
        self.R = R
        self.m = m
        self.A = np.array([
            [0, 1],
            [self.g / self.R, 0]
        ]) * (1 / self.dt)
        self.B = np.array([
            [0],
            [self.g / self.R]
        ]) * (1 / self.dt)
        self.x = np.array([0, 0]).reshape((2, 1))
        self.u = np.array([0]).reshape((1, 1))


class InvPendulumMPC:
    nx = 2
    nu = 1

    def __init__(self, model: InvertedPendulum2D, N: int, Q: Vector, Q_n: Vector, R: Vector):
        self.model: InvertedPendulum2D = model
        self.N: int = N  # predict horizon
        self.Q: Vector = Q  # penalization on difference to reference state in the predicted horizon
        self.Q_n: Vector = Q_n  # penalization on different to reference state on the final state
        self.R: Vector = R  # penalization on control, optimality
        self.ndes: int = (self.nx + self.nu) * self.N  # total number of decision variables

        assert self.Q.shape == (self.nx, self.nx)
        assert self.Q_n.shape == (self.nx, self.nx)
        assert self.R.shape == (self.nu, self.nu)

        self.solver = osqp.OSQP()

        self.z = np.zeros((self.ndes, 1))  # stacked decision var with dim ((nx + nu) * N, 1)
        self.H = np.zeros((self.ndes, self.ndes))
        for i in range(self.N):  # fill in Q matrices
            self.H[self.nx * i:self.nx * (i + 1), self.nx * i:self.nx * (i + 1)] = self.Q if i != self.N - 1 else self.Q_n
        for i in range(self.N):
            self.H[self.nx * self.N:self.ndes, self.nx * self.N:self.ndes] = block_diag(
                *[self.R for i in range(self.N)]
            )

        print(self.H)

        self.q = np.concatenate(
            [-Q]
        )


InvPendulumMPC(InvertedPendulum2D(10, 1), 10, np.diag([1, 2]), np.diag([1, 2]), np.diag([1]))
