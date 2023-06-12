from regulator import Regulator
import numpy as np
from numba import njit, jit


class IterativeLQR(Regulator):
    def __init__(self, model, cost, derivs):
        """Constructs an IterativeLQR solver.
        Args:
            dynamics: Plant dynamics.
            cost: Cost function.
        """
        print("iLQR")
        self.model = model
        self.cost = cost
        self.derivs = derivs
        self.max_regu = 10000
        self.min_regu = 0.01

    def calculate_control(self, x0, u_trj, max_iter=50, regulation_init=100):
        """Calculate control.
        Args:
            x0: Initial states.
            u_trj: Control trajectory.
            max_iter: Maximum iteration term to break early due to
                divergence. This can be disabled by setting it to None.
            regulation_init: The regularization is adapted based on whether 
                the new control and state trajectories improved the cost..
        """
        # First forward rollout states
        x_trj = self.model.rollout(x0, u_trj)
        total_cost = self.cost.cost_trj(x_trj, u_trj)

        # Setup traces
        cost_trace = [total_cost]
        redu_ratio_trace = [1]
        redu_trace = []
        regu_trace = [regulation_init]

        # Run main loop
        for it in range(max_iter):
            k_trj, K_trj, expected_cost_redu = backward_pass(
                x_trj, u_trj, regulation_init, self.derivs)
            x_trj_new, u_trj_new = forward_pass(
                x_trj, u_trj, k_trj, K_trj, self.model.discrete_dynamics)
            # Evaluate new trajectory, cost
            new_cost = self.cost.cost_trj(x_trj_new, u_trj_new)
            # Substract of first cost with new cost
            cost_redu = cost_trace[-1] - new_cost
            # Set ratio of reduction with expected reduction
            redu_ratio = cost_redu / abs(expected_cost_redu)
            if cost_redu > 0:
                # Accept new trajectories with lower regularization
                redu_ratio_trace.append(redu_ratio)
                cost_trace.append(new_cost)
                # Now make improvements
                x_trj = x_trj_new
                u_trj = u_trj_new
                regulation_init *= 0.7
            else:
                # Reject new trajectories and increase regularization init
                regulation_init *= 2.0
                cost_trace.append(cost_trace[-1])
                redu_ratio_trace.append(0)

            # find minimum
            regulation_init = min(
                max(regulation_init, self.min_regu), self.max_regu)
            regu_trace.append(regulation_init)
            regu_trace.append(cost_redu)

            # Early termination if expected improvement is small
            if expected_cost_redu <= 1e-6:
                break
            return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace

    def run_ilqr(self, x0, N, max_iter=50, regu_init=100, u_init=None):
        # First forward rollout
        if u_init is not None:
            u_trj = u_init
        else:
            u_trj = np.random.randn(N - 1, 2) * 0.0001
        x_trj = self.model.rollout(x0=x0, u_trj=u_trj)
        total_cost = self.cost.cost_trj(x_trj, u_trj)
        regu = regu_init
        max_regu = 10000
        min_regu = 0.01

        # Setup traces
        print("total_cost", total_cost)
        cost_trace = [total_cost]
        redu_ratio_trace = [1]
        redu_trace = []
        regu_trace = [regu]

        # Run main loop
        for it in range(max_iter):
            # Backward and forward pass
            k_trj, K_trj, expected_cost_redu = backward_pass(
                x_trj, u_trj, regu, self.derivs)
            x_trj_new, u_trj_new = forward_pass(
                x_trj, u_trj, k_trj, K_trj, self.model.discrete_dynamics)
            # Evaluate new trajectory
            total_cost = self.cost.cost_trj(x_trj_new, u_trj_new)
            print("iter:", it, total_cost)
            cost_redu = cost_trace[-1] - total_cost
            print("cost redu", cost_redu)
            redu_ratio = cost_redu / abs(expected_cost_redu)
            # Accept or reject iteration
            print("#"*20)
            if cost_redu > 0:
                print("!"*30)
                print("improvement")
                # Improvement! Accept new trajectories and lower regularization
                redu_ratio_trace.append(redu_ratio)
                cost_trace.append(total_cost)
                x_trj = x_trj_new
                u_trj = u_trj_new
                regu *= 0.7
            else:
                # Reject new trajectories and increase regularization
                regu *= 2.0
                print("Reject new trajectories")
                cost_trace.append(cost_trace[-1])
                redu_ratio_trace.append(0)
            regu = min(max(regu, min_regu), max_regu)
            regu_trace.append(regu)
            redu_trace.append(cost_redu)

            # Early termination if expected improvement is small
            if expected_cost_redu <= 1e-6:
                break

        return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace


@jit(forceobj=True)
def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    # Write checker if the shape are the same as initial
    Q_x = np.zeros(l_x.shape)
    Q_u = np.zeros(l_u.shape)
    Q_xx = np.zeros(l_xx.shape)
    Q_ux = np.zeros(l_ux.shape)
    Q_uu = np.zeros(l_uu.shape)
    # print(l_x.shape, V_x.T.shape, f_x.shape)
    # print(l_u.shape, V_x.T.shape, f_u.shape)
    # print(l_xx.shape, f_x.T.shape, V_xx.shape, f_x.shape)
    # print(l_ux.shape, f_u.T.shape, V_xx.shape, f_x.shape)
    # print(l_uu.shape, f_u.T.shape, V_xx.shape, f_x.shape)

    Q_x = l_x + V_x.T @ f_x
    # print(Q_x.shape, l_x.shape)
    Q_u = l_u + V_x.T @ f_u
    # print(Q_u.shape, l_u.shape)
    Q_xx = l_xx + f_x.T @ V_xx @ f_x
    # print(Q_xx.shape, l_xx.shape)
    Q_ux = l_ux + f_u.T @ V_xx @ f_x
    # print(Q_ux.shape, l_ux.shape)
    Q_uu = l_uu + f_u.T @ V_xx @ f_u
    # print(Q_uu.shape, l_uu.shape)
    return Q_x, Q_u, Q_xx, Q_ux, Q_uu


@jit(forceobj=True)
def gains(Q_uu, Q_u, Q_ux):
    Q_uu_inv = np.linalg.inv(Q_uu)
    k = -np.dot(Q_uu_inv, Q_u)
    K = -np.dot(Q_uu_inv, Q_ux)
    return k, K


@jit(forceobj=True)
def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    V_x = np.zeros(Q_x.shape)
    V_x = Q_x + Q_ux.T @ k + K.T @ Q_u + K.T @ Q_uu @ k
    V_xx = np.zeros(Q_xx.shape)
    V_xx = Q_xx + Q_ux.T @ K + K.T @ Q_ux + K.T @ Q_uu @ K
    return V_x, V_xx


@jit(forceobj=True)
def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))


@jit(forceobj=True)
def forward_pass(x_trj, u_trj, k_trj, K_trj, dynamics):
    """Forward_pass.
    At each timestamp the new updated control is applied to dynamics.
    Args:
        x_trj: State trajectory.
        u_trj: Control trajectory.
        k_trj: Gain traj small k.
        K_trj: Gain traj K
        dynamics: Dynamics function of model.
    Returns:
        x_trj_new: Updated state trajectory.
        u_trj_new: Updated control trajectory.
    """
    x_trj_new = np.zeros(x_trj.shape)
    x_trj_new[0, :] = x_trj[0, :]
    u_trj_new = np.zeros(u_trj.shape)
    for n in range(u_trj.shape[0]):
        u_trj_new[n, :] = u_trj[n, :] + k_trj[n, :] + \
            K_trj[n, :] @ (x_trj_new[n, :] - x_trj[n, :])
        x_trj_new[n+1, :] = dynamics(x_trj_new[n, :], u_trj_new[n, :])
    return x_trj_new, u_trj_new


@jit(forceobj=True)
def backward_pass(x_trj, u_trj, regu, derivs, expected_cost_redu=0):
    """Backward pass.
    Starts from terminal boundary condition
    Args:
        x_trj: State trajectory.
        u_trj: Control trajectory.
        regu: Gain traj small k.
        K_trj: Gain traj K
        dynamics: Dynamics function of model.
    Returns:
        x_trj_new: Updated state trajectory.
        u_trj_new: Updated control trajectory.
    """
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
    V_x = np.zeros((x_trj.shape[1],))
    V_xx = np.zeros((x_trj.shape[1], x_trj.shape[1]))
    # Terminal boundary condition
    V_x, V_xx = derivs.final(x_trj[-1, :])
    for n in range(u_trj.shape[0] - 1, -1, -1):
        # First compute derivatives, then the Q-terms
        l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivs.stage(
            x_trj[n, :], u_trj[n, :])
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(
            l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)

        # We add regularization to ensure that Q_uu is invertible and nicely conditioned
        Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0]) * regu
        k, K = gains(Q_uu_regu, Q_u, Q_ux)
        k_trj[n, :] = k
        K_trj[n, :, :] = K
        V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
        expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
    return k_trj, K_trj, expected_cost_redu
