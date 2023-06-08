import json
import numpy as np
import matplotlib.pyplot as plt


def read_path(filename="waypoints.json"):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['trajectory']


def telemetry(paths, dt=0.1, unwrap=False):
    paths = np.array(paths)
    dx, dy = np.diff(paths, axis=0).T
    speed = np.hypot(dx, dy)
    acceleration = np.diff(speed) / dt
    yaw = np.arctan2(dy, dx)
    dtyaw = np.diff(np.unwrap(yaw)) / dt
    if unwrap:
        yaw = np.unwrap(yaw)
    return paths, speed, acceleration, yaw, dtyaw


def subplot_path(paths, speed, acceleration, yaw, dyaw):
    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    # Plot paths
    axs[0, 0].plot(paths[:, 0], paths[:, 1], marker='o',
                   linestyle='-', color='blue')
    axs[0, 0].set_title("Paths")
    axs[0, 0].set_xlabel("X")
    axs[0, 0].set_ylabel("Y")
    axs[0, 0].grid(True)

    # Plot speed
    axs[1, 0].plot(speed, marker='o', linestyle='--', color='green')
    axs[1, 0].set_title("Speed")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Speed")
    axs[1, 0].grid(True)

    # Plot yaw
    axs[1, 1].plot(yaw, marker='o', linestyle='--', color='purple')
    axs[1, 1].set_title("Yaw Angle")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Yaw Angle (radians)")
    axs[1, 1].grid(True)

    # Plot acceleration
    axs[2, 0].plot(acceleration, marker='o', linestyle='--', color='red')
    axs[2, 0].set_title("Acceleration")
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].set_ylabel("Acceleration")
    axs[2, 0].grid(True)

    # Plot dyaw
    axs[2, 1].plot(dyaw, marker='o', linestyle='--', color='purple')
    axs[2, 1].set_title("dYaw Angle")
    axs[2, 1].set_xlabel("Time")
    axs[2, 1].set_ylabel("dYaw Angle (radians)")
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    paths = read_path()
    paths = paths[:5]
    paths, speed, acceleration, yaw, dtyaw = telemetry(
        paths, dt=0.1, unwrap=False)
    # subplot_path(paths, speed, acceleration, yaw, dtyaw)
    # print(paths.shape)
    # print(speed.shape)
    # print(acceleration.shape)
    # print(yaw.shape)
    # print(dtyaw)
    import numpy as np
    from model import AWsimModel
    from cost import CostPath
    from auto_derivatives import Derivatives
    from iLQR import IterativeLQR
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    model = AWsimModel()
    velocity = 20.0
    cost = CostPath(paths, velocity=velocity)
    derivs = Derivatives(model, cost)
    ilqr = IterativeLQR(model, cost, derivs)
    x0 = np.array([paths[0][0], paths[0][1], .0, .0])
    
    N = paths.shape[0]
    max_iter = 1000
    regu_init = 100
    x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = ilqr.run_ilqr(
        x0, N, max_iter, regu_init)
    print("xtr", x_trj)
    print("utr",u_trj)
    print("cost", cost_trace)
    subplot_path(paths, speed, acceleration, yaw, dtyaw)

    # Plot paths
    plt.figure(figsize=(8, 6))
    plt.plot(x_trj[:, 0], x_trj[:, 1], marker='o', linestyle='-', color='blue')
    plt.plot(paths[:, 0], paths[:, 1], marker='o', linestyle='-', color='red')
    plt.title("Paths")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
