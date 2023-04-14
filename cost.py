import numpy as np
import abc
from numba import njit


class Cost(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def cost_stage(self, x, u):
        """Computes cost to achieve stage.
        Args:
            x: states variables [state_size]
            u: control input [action_size]
        Returns:
            cost_stage: cost to go 
        """
        pass

    @abc.abstractmethod
    def cost_final(self, x):
        """Computes cost to achieve goal.
        Args:
            x: states variables [state_size]
            u: control input [action_size]
        Returns:
            cost_final: cost to goal
        """
        pass


class CostCircle(Cost):
    def __init__(self, r=2.0, velocity=2.0, eps=1e-6, dt=0.1):
        self.r = 2.0
        self.v_target = velocity
        self.eps = eps
        self.dt = dt

    def cost_stage(self, x, u):
        return _c_cost_stage(x, u, self.r, self.eps, self.v_target, self.dt)

    def cost_final(self, x):
        return _c_cost_final(x, self.r, self.eps, self.v_target)

    def cost_trj(self, x_trj, u_trj):
        """Computes the optimal controls.
        Args:
            x_trj: states variables: shape(N, n)
            u: control input: shape(N-1, m)
        Returns:
            total: sum of cost of stages and to reach goal
        """
        return _c_cost_traj(x_trj, u_trj, self.r, self.eps, self.v_target, self.dt)


@njit
def circle_pos(x, r, eps):
    c_circle = np.power(
        np.sqrt(np.power(x[0], 2) + np.power(x[1], 2) + eps) - r, 2)
    return c_circle


@njit
def circle_speed(x, v_target):
    c_speed = np.power(x[3] - v_target, 2)
    return c_speed


@njit
def circle_control(u, dt):
    c_speed = (np.power(u[0], 2) + np.power(u[1], 2)) * dt
    return c_speed


@njit
def _c_cost_stage(x, u, r, eps, v_target, dt):
    c_circle = circle_pos(x, r, eps)
    c_speed = circle_speed(x, v_target)
    c_control = circle_control(u, dt)
    return c_circle + c_speed + c_control


@njit
def _c_cost_final(x, r, eps, v_target):
    c_circle = circle_pos(x, r, eps)
    c_speed = circle_speed(x, v_target)
    return c_circle + c_speed


@njit
def _c_cost_traj(x_trj, u_trj, r, eps, v_target, dt):
    total = 0.0
    for n in range(u_trj.shape[0]):
        total += _c_cost_stage(x_trj[n], u_trj[n], r, eps, v_target, dt)
    total += _c_cost_final(x_trj[-1], r, eps, v_target)
    return total
