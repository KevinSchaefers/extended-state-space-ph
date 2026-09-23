"""Time integration methods used in Section 4.

The methods of Section 4.1 (Lotka-Volterra model) use the Gonzalez discrete gradient method and solve the arising nonlinear systems with ``scipy.optimize.root``.  The methods of Section 4.2 (LC-oscillator chain) use sparse direct LU factorizations and, where applicable, a simplified Newton iteration.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.optimize import root
from scipy.sparse.linalg import splu

DG_TOL = 1e-15  # relative tolerance of the nonlinear solver in Section 4.1
NEWTON_TOL = 1e-10  # relative tolerance of the simplified Newton iteration in Section 4.2


@dataclass
class SolverStats:
    """Computational effort of a time integration method, cf. Table 1.

    Attributes
    ----------
    steps : int
        Number of time steps.
    newton_iterations : int
        Total number of simplified Newton iterations.
    factorizations : int
        Total number of matrix factorizations.
    R_evaluations : int
        Total number of evaluations of the dissipation matrix.
    """

    steps: int = 0
    newton_iterations: int = 0
    factorizations: int = 0
    R_evaluations: int = 0

    @property
    def newton_iterations_per_step(self):
        return self.newton_iterations / self.steps

    @property
    def factorizations_per_step(self):
        return self.factorizations / self.steps

    @property
    def R_evaluations_per_step(self):
        return self.R_evaluations / self.steps


# --------------------------------------------------------------------------------------
# Section 4.1: discrete gradient methods on the extended state space
# --------------------------------------------------------------------------------------


def gonzalez_discrete_gradient(system, x, y):
    """Second-order Gonzalez discrete gradient of the Hamiltonian of ``system``."""
    dx = y - x
    norm_sq = np.dot(dx, dx)
    if norm_sq < 1e-14:
        return system.grad_H(x)
    grad_mid = system.grad_H((x + y) / 2)
    return grad_mid + dx * (system.H(y) - system.H(x) - np.dot(grad_mid, dx)) / norm_sq


def triple_jump_coefficients(order):
    """Coefficients ``(a, b)`` of the Strang splitting and its triple-jump compositions.

    Parameters
    ----------
    order : int
        Even order ``p``; ``order=2`` returns the coefficients of the Strang splitting, higher orders those of the iterated triple-jump compositions [Yoshida 1990].

    Returns
    -------
    (a, b) : arrays
        Coefficients of the ``v``- and ``w``-sub-steps.
    """
    a, b = [0.5, 0.5], [1.0, 0.0]
    for i in range(order // 2 - 1):
        gamma1 = 1 / (2 - 2 ** (1 / (2 * (i + 1) + 1)))
        gamma2 = 1 - 2 * gamma1
        a = np.concatenate([gamma1 * np.array(a), gamma2 * np.array(a), gamma1 * np.array(a)])
        b = np.concatenate([gamma1 * np.array(b), gamma2 * np.array(b), gamma1 * np.array(b)])
    return a, b


def _discrete_gradient_step(system, x, J_frozen, h):
    """One discrete gradient step for ``x' = J_frozen grad_H(x)``."""
    residual = lambda z: z - x - h * J_frozen @ gonzalez_discrete_gradient(system, x, z)
    return root(residual, x, method="hybr", tol=DG_TOL).x


def ess_strang_dg(system, x0, tspan, nsteps, order=2):
    """Extended state space method for the symmetric consistent approximations.

    Both numerical flows are Gonzalez discrete gradient methods, composed by the Strang splitting or, for ``order > 2``, by its triple-jump compositions.  The consistent approximations freeze ``J`` at the respective other copy of the state (choice (24) of the paper), such that both components are energy-conserving.

    Parameters
    ----------
    system : SkewGradientSystem
    x0 : (n,) array
        Initial value, used for both copies of the state.
    tspan : (2,) sequence
        Time interval.
    nsteps : int
        Number of time steps.
    order : int
        Even order of the underlying splitting method.

    Returns
    -------
    (t, v, w)
        Time grid and both components of the extended state.
    """
    h = (tspan[1] - tspan[0]) / nsteps
    a, b = triple_jump_coefficients(order)
    t = np.linspace(tspan[0], tspan[1], nsteps + 1)
    v = np.zeros((len(x0), nsteps + 1))
    w = np.zeros((len(x0), nsteps + 1))
    v[:, 0] = w[:, 0] = x0
    # Trial iterates of the nonlinear solver may leave the domain of the logarithm.
    with np.errstate(invalid="ignore", divide="ignore"):
        for n in range(nsteps):
            vn, wn = v[:, n].copy(), w[:, n].copy()
            for ai, bi in zip(a, b):
                if ai != 0:
                    vn = _discrete_gradient_step(system, vn, system.J(wn), ai * h)
                if bi != 0:
                    wn = _discrete_gradient_step(system, wn, system.J(vn), bi * h)
            v[:, n + 1], w[:, n + 1] = vn, wn
    return t, v, w


def ess_strang_dg_one_sided(system, x0, tspan, nsteps):
    """Extended state space method for the one-sided consistent approximations.

    The consistent approximations evaluate the effort function at ``v`` in both subsystems (choice (21) of the paper), such that only the component ``v`` satisfies a power balance.  The vector field of the ``w``-subsystem is constant, so its flow is computed exactly; the ``v``-subsystem is solved by the Gonzalez discrete gradient method.  The Hamiltonian is never evaluated at ``w``.

    Parameters
    ----------
    system : SkewGradientSystem
    x0 : (n,) array
        Initial value, used for both copies of the state.
    tspan : (2,) sequence
        Time interval.
    nsteps : int
        Number of time steps.

    Returns
    -------
    (t, v, w)
        Time grid and both components of the extended state.
    """
    h = (tspan[1] - tspan[0]) / nsteps
    t = np.linspace(tspan[0], tspan[1], nsteps + 1)
    v = np.zeros((len(x0), nsteps + 1))
    w = np.zeros((len(x0), nsteps + 1))
    v[:, 0] = w[:, 0] = x0
    with np.errstate(invalid="ignore", divide="ignore"):
        for n in range(nsteps):
            vn, wn = v[:, n], w[:, n]
            vn = _discrete_gradient_step(system, vn, system.J(wn), h / 2)
            wn = wn + h * system.J(vn) @ system.grad_H(vn)
            vn = _discrete_gradient_step(system, vn, system.J(wn), h / 2)
            v[:, n + 1], w[:, n + 1] = vn, wn
    return t, v, w


# --------------------------------------------------------------------------------------
# Section 4.2: methods for the LC-oscillator chain
# --------------------------------------------------------------------------------------


def simplified_newton(residual, lu, y0, stats, tol=NEWTON_TOL, maxit=200):
    """Simplified Newton iteration with a fixed, pre-factorized iteration matrix.

    The iteration terminates once the relative change between consecutive iterates
    falls below ``tol``.
    """
    y = y0.copy()
    for it in range(1, maxit + 1):
        dy = lu.solve(-residual(y))
        y += dy
        if np.linalg.norm(dy) <= tol * np.linalg.norm(y):
            stats.newton_iterations += it
            return y
    raise RuntimeError("simplified Newton iteration did not converge")


def gl1(system, x0, tspan, nsteps):
    """1-stage Gauss-Legendre method (implicit midpoint rule).

    The iteration matrix freezes ``R`` at the beginning of the time step and is
    factorized once per time step; the residual evaluates ``R`` at the current iterate.

    Returns
    -------
    (t, x, stats)
    """
    h = (tspan[1] - tspan[0]) / nsteps
    E, J = system.E, system.J
    t = np.linspace(tspan[0], tspan[1], nsteps + 1)
    x = np.zeros((len(x0), nsteps + 1))
    x[:, 0] = x0
    stats = SolverStats(steps=nsteps)
    for k in range(nsteps):
        xk = x[:, k]
        Bu = system.input_term(t[k] + h / 2)

        def residual(y, xk=xk, Bu=Bu):
            stats.R_evaluations += 1
            return E @ (y - xk) - h * ((J - system.R((xk + y) / 2)) @ ((xk + y) / 2) + Bu)

        lu = splu((E - h / 2 * (J - system.R(xk))).tocsc())
        stats.factorizations += 1
        stats.R_evaluations += 1
        x[:, k + 1] = simplified_newton(residual, lu, xk, stats)
    return t, x, stats


def gl2(system, x0, tspan, nsteps):
    """2-stage Gauss-Legendre method of order four.

    The two stages are solved simultaneously; the iteration matrix freezes ``R`` at
    the beginning of the time step and is factorized once per time step.

    Returns
    -------
    (t, x, stats)
    """
    h = (tspan[1] - tspan[0]) / nsteps
    E, J = system.E, system.J
    d = len(x0)
    s3 = np.sqrt(3)
    A = np.array([[1 / 4, 1 / 4 - s3 / 6], [1 / 4 + s3 / 6, 1 / 4]])
    b = np.array([1 / 2, 1 / 2])
    c = np.array([1 / 2 - s3 / 6, 1 / 2 + s3 / 6])
    t = np.linspace(tspan[0], tspan[1], nsteps + 1)
    x = np.zeros((d, nsteps + 1))
    x[:, 0] = x0
    stats = SolverStats(steps=nsteps)
    for k in range(nsteps):
        xk = x[:, k]
        Bu = [system.input_term(t[k] + c[i] * h) for i in range(2)]

        def residual(z, xk=xk, Bu=Bu):
            K = z.reshape(2, d)
            X = [xk + h * (A[i, 0] * K[0] + A[i, 1] * K[1]) for i in range(2)]
            stats.R_evaluations += 2
            return np.concatenate(
                [E @ K[i] - ((J - system.R(X[i])) @ X[i] + Bu[i]) for i in range(2)]
            )

        A0 = J - system.R(xk)
        stats.R_evaluations += 1
        M = sp.bmat(
            [[E - h * A[i, j] * A0 if i == j else -h * A[i, j] * A0 for j in range(2)] for i in range(2)],
            format="csc",
        )
        lu = splu(M)
        stats.factorizations += 1
        K0 = np.tile(system.rhs(t[k], xk), 2)
        K = simplified_newton(residual, lu, K0, stats).reshape(2, d)
        x[:, k + 1] = xk + h * (b[0] * K[0] + b[1] * K[1])
    return t, x, stats


def _expm_Einv_J(system, tau):
    """``exp(tau E^{-1} J)``, exploiting the 2x2 block structure of ``E^{-1} J``."""
    Ed = system.E_diag
    blocks = [
        expm(tau * np.array([[0.0, -1 / Ed[2 * i]], [1 / Ed[2 * i + 1], 0.0]]))
        for i in range(len(Ed) // 2)
    ]
    return sp.block_diag(blocks, format="csr")


def ead(system, x0, tspan, nsteps):
    """Energy-associated splitting method of Section 4.2.

    Strang splitting of the energy-conservative and the passive part.  The flow of the conservative part is computed exactly via the matrix exponential, which is precomputed once; the passive part is solved by the 1-stage Gauss-Legendre method (implicit midpoint rule).

    Returns
    -------
    (t, x, stats)
    """
    h = (tspan[1] - tspan[0]) / nsteps
    E = system.E
    Phi_J = _expm_Einv_J(system, h / 2)
    t = np.linspace(tspan[0], tspan[1], nsteps + 1)
    x = np.zeros((len(x0), nsteps + 1))
    x[:, 0] = x0
    stats = SolverStats(steps=nsteps)
    for k in range(nsteps):
        y0 = Phi_J @ x[:, k]
        Bu = system.input_term(t[k] + h / 2)

        def residual(y, y0=y0, Bu=Bu):
            stats.R_evaluations += 1
            return E @ (y - y0) + h * (system.R((y0 + y) / 2) @ ((y0 + y) / 2)) - h * Bu

        lu = splu((E + h / 2 * system.R(y0)).tocsc())
        stats.factorizations += 1
        stats.R_evaluations += 1
        x[:, k + 1] = Phi_J @ simplified_newton(residual, lu, y0, stats)
    return t, x, stats


def ess(system, x0, tspan, nsteps):
    """Extended state space method of Section 4.2.

    Strang splitting on the extended state space with the consistent approximations that freeze the structure matrices at the respective other copy of the state (choice (24) of the paper). Both sub-flows are 1-stage Gauss-Legendre methods (implicit midpoint rule) and reduce to linear systems, so no nonlinear iteration is required.  The FSAL property is exploited: the factorization of the last ``v``-sub-step is reused in the next time step.

    Returns
    -------
    (t, v, w, stats)
        Time grid, both components of the extended state, and the effort statistics.
    """
    h = (tspan[1] - tspan[0]) / nsteps
    E, J = system.E, system.J
    t = np.linspace(tspan[0], tspan[1], nsteps + 1)
    v = np.zeros((len(x0), nsteps + 1))
    w = np.zeros((len(x0), nsteps + 1))
    v[:, 0] = w[:, 0] = x0
    stats = SolverStats(steps=nsteps)

    A_w = J - system.R(x0)
    stats.R_evaluations += 1
    lu_v = splu((E - h / 4 * A_w).tocsc())
    stats.factorizations += 1
    P_v = E + h / 4 * A_w

    for k in range(nsteps):
        # v half step, structure matrix frozen at w (factorization from the previous step)
        v_half = lu_v.solve(P_v @ v[:, k] + h / 2 * system.input_term(t[k]))

        # w full step, structure matrix frozen at v
        A_v = J - system.R(v_half)
        stats.R_evaluations += 1
        lu_w = splu((E - h / 2 * A_v).tocsc())
        stats.factorizations += 1
        w[:, k + 1] = lu_w.solve((E + h / 2 * A_v) @ w[:, k] + h * system.input_term(t[k] + h / 2))

        # v half step, structure matrix frozen at the new w (reused in the next step)
        A_w = J - system.R(w[:, k + 1])
        stats.R_evaluations += 1
        lu_v = splu((E - h / 4 * A_w).tocsc())
        stats.factorizations += 1
        P_v = E + h / 4 * A_w
        v[:, k + 1] = lu_v.solve(P_v @ v_half + h / 2 * system.input_term(t[k] + h))
    return t, v, w, stats
