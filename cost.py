import jax.numpy as jnp
import abc
from numba import njit, jit


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


class CostPath(Cost):
    def __init__(self, path, velocity=.2, eps=1e-6, dt=0.1):
        self.r = 2.0
        self.v_target = velocity
        self.eps = eps
        self.dt = dt
        self.path = path
        self.ref_x = 0
        self.ref_y = 0
        self.ref_pose = jnp.zeros(2)

    def straight_pos(self, x, eps):
        diff = x[:2] - self.ref_pose
        c_straight = jnp.sqrt(jnp.square(diff))
        return c_straight

    def cost_stage(self, x, u):
        c_straight = self.straight_pos(x, self.eps)
        c_speed = cost_speed(x, self.v_target)
        c_control = cost_control(u, self.dt)
        return c_straight + c_speed + c_control

    def cost_final(self, x):
        c_straight = self.straight_pos(x, self.eps)
        c_speed = cost_speed(x, self.v_target)
        return c_straight + c_speed

    @jit(forceobj=True)
    def cost_trj(self, x_trj, u_trj):
        """Computes the optimal controls.
        Args:
            x_trj: states variables: shape(N, n)
            u: control input: shape(N-1, m)
        Returns:
            total: sum of cost of stages and to reach goal
        """
        total = 0.0
        for n in range(u_trj.shape[0]):
            self.ref_x, self.ref_y = self.path[n]
            self.ref_pose = self.path[n]
            # print("ref",  self.path[n].shape,  self.path[n])
            # print("x_tr", x_trj[n][:2].shape, x_trj[n][:2])
            total += self.cost_stage(x_trj[n], u_trj[n])
        total += self.cost_final(x_trj[-1])
        return total


class CostCircle(Cost):
    def __init__(self, r=2.0, velocity=2.0, eps=1e-6, dt=0.1):
        self.r = 2.0
        self.v_target = velocity
        self.eps = eps
        self.dt = dt

    def cost_stage(self, x, u):
        c_circle = circle_pos(x, self.r, self.eps)
        c_speed = cost_speed(x, self.v_target)
        c_control = cost_control(u, self.dt)
        return c_circle + c_speed + c_control

    def cost_final(self, x):
        c_circle = circle_pos(x, self.r, self.eps)
        c_speed = cost_speed(x, self.v_target)
        return c_circle + c_speed

    @jit(forceobj=True)
    def cost_trj(self, x_trj, u_trj):
        """Computes the optimal controls.
        Args:
            x_trj: states variables: shape(N, n)
            u: control input: shape(N-1, m)
        Returns:
            total: sum of cost of stages and to reach goal
        """
        total = 0.0
        for n in range(u_trj.shape[0]):
            total += self.cost_stage(x_trj[n], u_trj[n])
        total += self.cost_final(x_trj[-1])
        return total


@jit(forceobj=True)
def circle_pos(x, r, eps):
    c_circle = jnp.power(
        jnp.sqrt(jnp.power(x[0], 2) + jnp.power(x[1], 2) + eps) - r, 2)
    return c_circle


@jit(forceobj=True)
def cost_speed(x, v_target):
    c_speed = jnp.power(x[3] - v_target, 2)
    return c_speed


@jit(forceobj=True)
def cost_control(u, dt):
    c_speed = (jnp.power(u[0], 2) + jnp.power(u[1], 2)) * dt
    return c_speed


@jit(forceobj=True)
def straight_pos(x, ref, eps):
    c_straight = jnp.power(
        jnp.sqrt(jnp.power(x[0] - ref[0], 2) + jnp.power(x[1] - ref[1], 2) + eps), 2)
    return c_straight
