import enum

import numpy as np
import qpsolvers
from qpsolvers import Solution
from scipy.linalg import block_diag
from scipy import sparse
from dataclasses import dataclass

from typing import List, Callable
from common_types import *


class TaskType(enum.Enum):
    EMPTY = 0
    COST = 1
    EQUALITY_CONSTRAINT = 2
    INEQUALITY_CONSTRAINT = 3
    LB_UB = 4


class Task(object):
    """
    Define a task to be solved inside a large QP formulation.
    It follows the following qp formulation:

    min_x   1/2 * x.T @ P @ x + q.T @ x
    s.t.    A @ x = b
            G @ x <= h
            lb <= x <= ub

    Task has 4 typical types, corresponding to cost, equality, inequality and lb/ub constraints.
    """

    P: Matrix
    q: Vector
    A: Matrix
    b: Vector
    G: Matrix
    h: Vector
    lb: Vector
    ub: Vector
    task_type = TaskType.EMPTY

    def __init__(
            self, task_type: TaskType,
            P=None, q=None,
            A=None, b=None,
            G=None, h=None,
            Lb=None, Ub=None
    ):
        self.task_type = task_type
        self.P = P
        self.q = q
        self.A = A
        self.b = b
        self.G = G
        self.h = h
        self.lb = Lb
        self.ub = Ub

    def check(self) -> None:
        if self.task_type == TaskType.COST:
            assert self.P is not None and self.q is not None, "Cost task must have P and q."
        if self.task_type == TaskType.EQUALITY_CONSTRAINT:
            assert self.A is not None and self.b is not None, "Equality task must have A and b."
        if self.task_type == TaskType.INEQUALITY_CONSTRAINT:
            assert self.G is not None and self.h is not None, "Inequality task must have G and h."
        if self.task_type == TaskType.LB_UB:
            assert self.lb is not None and self.ub is not None, "Lb/Ub task must have Lb and Ub."


class QPFormulator(object):
    def __init__(self, n_des: int):
        """
        Initialize the QP formulator.
        @param n_des: number of decision variables.
        """
        self.solver = None
        self.n_des = n_des
        self.n_ueq_constraint = 0
        self.n_eq_constraint = 0
        self.bounded = False

        # QP Problem and Tasks
        self.problem: qpsolvers.Problem = qpsolvers.Problem(
            sparse.csc_matrix((self.n_des, self.n_des)),
            np.zeros((self.n_des, 1))
        )
        self.tasks: List[Task] = []

    def add_task(self, task: Task) -> None:
        if task is None or task.task_type == TaskType.EMPTY:
            return
        if task.task_type == TaskType.COST:
            assert task.P.shape == (self.n_des, self.n_des)
            assert task.q.shape == (self.n_des,)
        if task.task_type == TaskType.EQUALITY_CONSTRAINT:
            assert task.A.shape[1] == self.n_des
            assert task.b.shape[0] == task.A.shape[0]
            self.n_eq_constraint += task.A.shape[0]
        if task.task_type == TaskType.INEQUALITY_CONSTRAINT:
            assert task.G.shape[1] == self.n_des
            assert task.h.shape[0] == task.G.shape[0]
            self.n_ueq_constraint += task.G.shape[0]
        if task.task_type == TaskType.LB_UB:
            assert task.lb.shape[0] == task.ub.shape[0]
            self.bounded = True

        self.tasks.append(task)

    def summary(self) -> None:
        print(
            f"""
            ------- QP Formulator -------
            Solving for QP Problem:
                # Decision Variables: {self.n_des}
                # Inequality Constraints: {self.n_ueq_constraint}
                # Equality Constraints: {self.n_eq_constraint}
                # Bounded: {self.bounded}
            """
        )

    def solve(self) -> Solution | None:
        self.problem.P = sparse.csc_matrix((self.n_des, self.n_des))  # shape: n * n
        self.problem.q = np.zeros(self.n_des)  # shape: 1 * n
        self.problem.lb = None
        self.problem.ub = None
        A_s = []
        b_s = []
        G_s = []
        h_s = []

        for task in self.tasks:
            if task.task_type == TaskType.EMPTY:
                continue
            if task.task_type == TaskType.COST:
                self.problem.P += sparse.csc_matrix(task.P)
                self.problem.q += task.q
            if task.task_type == TaskType.EQUALITY_CONSTRAINT:
                A_s.append(task.A)
                b_s.append(task.b)
            if task.task_type == TaskType.INEQUALITY_CONSTRAINT:
                G_s.append(task.G)
                h_s.append(task.h)
            if task.task_type == TaskType.LB_UB:
                if self.problem.lb is None or self.problem.ub is None:
                    self.problem.lb = task.lb
                    self.problem.ub = task.ub
                else:
                    self.problem.lb = np.min(task.lb, self.problem.lb)
                    self.problem.ub = np.max(task.ub, self.problem.ub)

        self.problem.A = np.vstack(A_s) if len(A_s) != 0 else None
        self.problem.b = np.hstack(b_s) if len(b_s) != 0 else None
        self.problem.G = np.vstack(G_s) if len(G_s) != 0 else None
        self.problem.h = np.hstack(h_s) if len(h_s) != 0 else None


        try:
            self.problem.check_constraints()
        except qpsolvers.ProblemError as e:
            print(f"Constraint check failed! {e}")
            return None

        solution = qpsolvers.solve_problem(self.problem, solver="osqp", polish=True, check_termination=5)
        return solution
