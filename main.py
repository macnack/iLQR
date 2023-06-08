import numpy as np
from model import CarModel
from cost import CostCircle
from auto_derivatives import Derivatives
from iLQR import IterativeLQR
import matplotlib.pyplot as plt
import matplotlib as mpl
n_x = 5
n_u = 2

car = CarModel()

N = 10
x0 = np.array([1, 0, 0, 1, 0])
u_trj = np.zeros((N - 1, n_u))
x_trj = car.rollout(x0, u_trj)

circe = CostCircle()
derivs = Derivatives(car, circe)
ilqr = IterativeLQR(car,circe, derivs)

x0 = np.array([-3.0, 1.0, -0.2, 0.0, 0.0])
N = 50
max_iter = 50
regu_init = 100
# circe.cost_trj(x_trj, u_trj)
x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = ilqr.run_ilqr(
    x0, N, max_iter, regu_init
)


r = 2.0
v_target = 2.0
eps = 1e-6  # The derivative of sqrt(x) at x=0 is undefined. Avoid by subtle smoothing

plt.figure(figsize=(9.5, 8))
# Plot circle
theta = np.linspace(0, 2 * np.pi, 100)
plt.plot(r * np.cos(theta), r * np.sin(theta), linewidth=5)
ax = plt.gca()

# Plot resulting trajecotry of car
plt.plot(x_trj[:, 0], x_trj[:, 1], linewidth=5)
w = 2.0
h = 1.0

# Plot rectangles
for n in range(x_trj.shape[0]):
    rect = mpl.patches.Rectangle((-w / 2, -h / 2), w, h, fill=False)
    t = (
        mpl.transforms.Affine2D()
        .rotate_deg_around(0, 0, np.rad2deg(x_trj[n, 2]))
        .translate(x_trj[n, 0], x_trj[n, 1])
        + ax.transData
    )
    rect.set_transform(t)
    ax.add_patch(rect)
ax.set_aspect(1)
plt.ylim((-3, 3))
plt.xlim((-4.5, 3))
plt.tight_layout()
plt.show()

# plt.subplots(figsize=(10, 6))
# Plot results
plt.subplot(2, 2, 1)
plt.plot(cost_trace)
plt.xlabel("# Iteration")
plt.ylabel("Total cost")
plt.title("Cost trace")

plt.subplot(2, 2, 2)
delta_opt = np.array(cost_trace) - cost_trace[-1]
plt.plot(delta_opt)
plt.yscale("log")
plt.xlabel("# Iteration")
plt.ylabel("Optimality gap")
plt.title("Convergence plot")

plt.subplot(2, 2, 3)
plt.plot(redu_ratio_trace)
plt.title("Ratio of actual reduction and expected reduction")
plt.ylabel("Reduction ratio")
plt.xlabel("# Iteration")

plt.subplot(2, 2, 4)
plt.plot(regu_trace)
plt.title("Regularization trace")
plt.ylabel("Regularization")
plt.xlabel("# Iteration")
plt.tight_layout()
plt.show()
