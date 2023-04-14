import numpy as np


class CarModel:
    def __init__(self):
        # add number of states and number of inputs
        # avoid wrong shapes
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

    def discrete_dynamics(self, x, u, dt=0.1):
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
        return x + dt * self.continuous_dynamics(x, u)

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
