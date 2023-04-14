import jax.numpy as jnp
from jax import jacfwd, hessian, jit
from jax import random


class Derivatives:
    def __init__(self, Model, CostClass):
        # Cost function
        l = CostClass.cost_stage
        # Jacobian [dl/dx dl/du]
        self.jac_l = jit(jacfwd(l, argnums=(0, 1)))
        # Hessian [dl/dx^2 dl/du^2]
        self.hes_l = jit(hessian(l, argnums=(0, 1)))

        # Final cost
        l_final = CostClass.cost_final
        # Jacobian [dlf/dx dlf/du]
        self.jac_lf = jit(jacfwd(l_final))
        # Hessian [dlf/dx^2 dlf/du^2]
        self.hes_lf = jit(hessian(l_final))

        # Dynamic function
        f = Model.discrete_dynamics
        # Jacobian f
        self.jac_f = jit(jacfwd(f, argnums=(0, 1)))

    def stage(self, x, u):
        l_x, l_u = self.jac_l(x, u)
        (l_xx, _), (l_ux, l_uu) = self.hes_l(x, u)
        f_x, f_u = self.jac_f(x, u)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

    def final(self, x):
        l_final_x = self.jac_lf(x)
        l_final_xx = self.hes_lf(x)

        return l_final_x, l_final_xx
