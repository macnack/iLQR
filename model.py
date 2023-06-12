import numpy as np
import jax.numpy as jnp


class AWsimModel:
    def __init__(self, dt = 0.1):
        # add number of states and number of inputs
        # avoid wrong shapes
        print("AWSIM model")

    def discrete_dynamics(self, x, u, dt=0.1):
        """Discrete model
        Args:
            x: states variables [state_size]
                [x position, y position, heading, speed]
            u: control input [action_size]
                [acceleration, steering]
            dt: sampling time
        Returns:
            Next state [state_size].
        """
        heading = jnp.asarray(x)[2]
        v = jnp.asarray(x)[3]

        acceleration = jnp.asarray(u)[0]
        steering = jnp.asarray(u)[1]
        acceleration = np.clip(acceleration, 0.01, 0.2)
        steering = np.clip(steering, -1.0, 1.0)

        x_next = jnp.array(
            [v * jnp.cos(heading), v * jnp.sin(heading),
             v * jnp.tan(steering) / 0.28, acceleration]
        )
        return x + dt * x_next

    def rollout(self, x0, u_trj):
        """Rollout trajectory
        Args:
            x0: initial states [state_size]
                [x position, y position, heading, speed]
            u_trj: control trajectory: shape [N, number of states]
                [acceleration, steering]
            dt: sampling time
        Returns:
            x_trj: state trajectory: shape[N+1, number of states].
        """
        x_trj = np.zeros((u_trj.shape[0] + 1, x0.shape[0]))
        x_trj[0] = x0
        for n in range((u_trj.shape[0])):
            x_trj[n+1] = self.discrete_dynamics(x_trj[n], u_trj[n])
        return x_trj


class CarModel:
    def __init__(self, dt=0.1):
        # add number of states and number of inputs
        # avoid wrong shapes
        self.dt = dt
        print("Car model")

    def continuous_dynamics(self, x, u):
        """Continuous model
        Args:
            x: states variables [state_size]
                [x position, y position, heading, speed, steering angle]
            u: control input [action_size]
                [acceleration, steering velocity]
        Returns:
            Next state [state_size].
        """
        heading = x[2]
        v = x[3]
        steer = x[4]
        x_next = np.array(
            [v * np.cos(heading), v * np.sin(heading),
             v * np.tan(steer), u[0], u[1]]
        )
        return x_next

    def discrete_dynamics(self, x, u):
        """Discrete model
        Args:
            x: states variables [state_size]
                [x position, y position, heading, speed, steering angle]
            u: control input [action_size]
                [acceleration, steering velocity]
            dt: sampling time
        Returns:
            Next state [state_size].
        """
        heading = jnp.asarray(x)[2]
        v = jnp.asarray(x)[3]
        steer = jnp.asarray(x)[4]
        acceleration = jnp.asarray(u)[0]
        # acceleration = jnp.clip(acceleration, 0.01, 0.2)
        sterring = jnp.asarray(u)[1]
        # sterring = jnp.clip(sterring, -1.0, 1.0)
        x_next = jnp.array(
            [v * jnp.cos(heading), v * jnp.sin(heading),
             v * jnp.tan(steer), acceleration, sterring]
        )
        return x + self.dt * x_next

    def rollout(self, x0, u_trj):
        """Rollout trajectory
        Args:
            x0: initial states [state_size]
                [x position, y position, heading, speed, steering angle]
            u_trj: control trajectory: shape [N, number of states]
                [acceleration, steering velocity]
            dt: sampling time
        Returns:
            x_trj: state trajectory: shape[N+1, number of states].
        """
        x_trj = np.zeros((u_trj.shape[0] + 1, x0.shape[0]))
        x_trj[0] = x0
        for n in range((u_trj.shape[0])):
            x_trj[n+1] = self.discrete_dynamics(x_trj[n], u_trj[n])
        return x_trj
