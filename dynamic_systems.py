from abc import abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from baxter import BaxterParameters


@dataclass
class IntegralSolution:
    solution: np.ndarray = None
    n_iter: int = None


class DynamicSystem:

    @property
    def n_state(self):
        raise NotImplementedError()

    @property
    def n_input(self):
        raise NotImplementedError()

    @abstractmethod
    def dynamic(self, Z_t, t, U_delay):
        ...

    @abstractmethod
    def kappa(self, Z_t, t):
        ...


class Unicycle(DynamicSystem):
    @property
    def n_state(self):
        return 3

    @property
    def n_input(self):
        return 2

    def dynamic(self, Z_t, t, U_delay):
        x, y, theta = Z_t
        omega, nu = U_delay
        x_dot = nu * np.cos(theta)
        y_dot = nu * np.sin(theta)
        theta_dot = omega
        return np.array([x_dot, y_dot, theta_dot])

    def kappa(self, Z_t, t):
        x, y, theta = Z_t
        p = x * np.cos(theta) + y * np.sin(theta)
        q = x * np.sin(theta) - y * np.cos(theta)
        omega = -5 * p ** 2 * np.cos(3 * t) - p * q * (1 + 25 * np.cos(3 * t) ** 2) - theta
        nu = -p + 5 * q * (np.sin(3 * t) - np.cos(3 * t)) + q * omega
        return np.array([omega, nu])


class Baxter(DynamicSystem):

    @property
    def n_input(self):
        return self.dof

    @property
    def n_state(self):
        return self.dof * 2  # dof dimensions for e1 and dof dimensions for e2

    def __init__(self, alpha=1, beta=1, dof: int = 7, f: float = 0.1, magnitude: float = 0.2):
        assert 1 <= dof <= 7
        self.dof = dof
        self.f = f
        self.magnitude = magnitude
        self.alpha = alpha * np.eye(dof)
        self.beta = beta * np.eye(dof)
        self.baxter_parameters = BaxterParameters(dof=dof)

    @lru_cache(maxsize=None)
    def G(self, t):
        return self.baxter_parameters.compute_gravity_vector(self.q_des(t))

    @lru_cache(maxsize=None)
    def C(self, t):
        return self.baxter_parameters.compute_coriolis_centrifugal_matrix(self.q_des(t), self.qd_des(t))

    @lru_cache(maxsize=None)
    def M(self, t):
        return self.baxter_parameters.compute_inertia_matrix(self.q_des(t))

    @lru_cache(maxsize=None)
    def q_des(self, t):
        return self.magnitude * np.array(
            [np.sin(self.f * t), np.cos(self.f * t), np.sin(self.f * t), np.cos(self.f * t), np.sin(self.f * t),
             np.cos(self.f * t), np.sin(self.f * t)])[:self.dof]

    @lru_cache(maxsize=None)
    def qd_des(self, t):
        return self.magnitude * self.f * np.array(
            [np.cos(self.f * t), - np.sin(self.f * t), np.cos(self.f * t), -np.sin(self.f * t), np.cos(self.f * t),
             - np.sin(self.f * t), np.cos(self.f * t)]
        )[:self.dof]

    @lru_cache(maxsize=None)
    def qdd_des(self, t):
        return self.magnitude * -self.f ** 2 * np.array(
            [np.sin(self.f * t), np.cos(self.f * t), np.sin(self.f * t), np.cos(self.f * t), np.sin(self.f * t),
             np.cos(self.f * t), np.sin(self.f * t)]
        )[:self.dof]

    def h(self, e1, e2, t):
        return self.qdd_des(t) - self.alpha @ (self.alpha @ e1) + np.linalg.inv(self.M(t)) @ (
                self.C(t) @ self.qd_des(t)
                + self.C(t) @ (self.alpha @ e1) - self.C(t) @ e2)

    def q(self, E_t, t):
        e1_t, e2_t = E_t[:self.dof], E_t[self.dof:]
        q = self.q_des(t) - e1_t
        return q

    def dynamic(self, E_t, t, U_delay):
        e1_t, e2_t = E_t[:self.dof], E_t[self.dof:]
        h = self.h(e1_t, e2_t, t)

        e1_t_dot = e2_t - self.alpha @ e1_t
        e2_t_dot = self.alpha @ e2_t + h - np.linalg.inv(self.M(t)) @ U_delay
        return np.concatenate([e1_t_dot, e2_t_dot])

    def kappa(self, E_t, t):
        e1, e2 = E_t[:self.dof], E_t[self.dof:]
        h = self.h(e1, e2, t)
        return self.M(t) @ (h + (self.beta + self.alpha) @ e2)


def model_forward(model, U_D, Z_t, t):
    device = next(model.parameters()).device
    u_tensor = torch.tensor(U_D, dtype=torch.float32, device=device).unsqueeze(0)
    z_tensor = torch.tensor(Z_t, dtype=torch.float32, device=device).unsqueeze(0)
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0)
    outputs = model(
        **{
            't': t_tensor,
            'z': z_tensor,
            'u': u_tensor,
            'label': None,
            'input': None
        }
    )
    return outputs.to('cpu').detach().numpy()[0]


def solve_integral(Z_t, P_D, U_D, t: float, dataset_config):
    assert len(P_D) == len(U_D)
    system = dataset_config.system
    dt = dataset_config.dt

    n_state = system.n_state
    n_points = len(P_D)

    ts = np.linspace(t, t + n_points * dt - dt, n_points)

    res, n_iter = solve_integral_successive(Z_t=Z_t, n_points=n_points, n_state=n_state, dt=dt, U_D=U_D,
                                            f=system.dynamic,
                                            threshold=dataset_config.successive_approximation_threshold, ts=ts)
    return IntegralSolution(solution=res, n_iter=n_iter)


def solve_integral_successive(Z_t, n_points: int, n_state: int, dt: float, U_D: np.ndarray, f, ts: np.ndarray,
                              n_iterations: int = 1, threshold: float = 1e-5):
    assert n_iterations >= 0
    assert isinstance(Z_t, np.ndarray)
    P_D = np.zeros((2, n_points + 1, n_state))
    P_D[0, :, :] = Z_t
    P_D[1, :, :] = Z_t

    n_iterations = 0
    while True:
        n_iterations += 1
        for j, t in enumerate(ts):
            if j == 0:
                P_D[1, j + 1, :] = Z_t + dt * f(P_D[0, 0, :], t, U_D[0])
            else:
                P_D[1, j + 1, :] = P_D[1, j, :] + dt * f(P_D[0, j, :], t, U_D[j])

        if np.all(np.abs(P_D[1] - P_D[0]) < threshold) or n_iterations > 100:
            break

        P_D[0, :, :] = P_D[1, :, :]

    return P_D[1, -1, :], n_iterations


if __name__ == '__main__':
    ...
