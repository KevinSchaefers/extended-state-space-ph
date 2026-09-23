"""System classes and the two benchmark problems of Section 4."""

import numpy as np
import scipy.sparse as sp


class SkewGradientSystem:
    """Skew-gradient system ``x' = J(x) grad_H(x)`` with point-wise skew-symmetric ``J``.

    Parameters
    ----------
    J : callable
        ``J(x) -> (n, n) array``, point-wise skew-symmetric.
    H : callable
        Hamiltonian ``H(x) -> float``.
    grad_H : callable
        Gradient of the Hamiltonian, ``grad_H(x) -> (n,) array``.

    Attributes
    ----------
    rhs : callable
        Right-hand side ``rhs(t, x) -> (n,) array`` in explicit form.
    """

    def __init__(self, J, H, grad_H):
        self.J = J
        self.H = H
        self.grad_H = grad_H

    def rhs(self, t, x):
        return self.J(x) @ self.grad_H(x)


class pHSystem:
    """Port-Hamiltonian ODE ``E x' = (J - R(x)) x + B u(t)`` with constant diagonal ``E``.

    Parameters
    ----------
    E_diag : (n,) array
        Diagonal of the flow matrix ``E``.
    J : (n, n) sparse matrix
        Constant interconnection matrix.
    R : callable
        ``R(x) -> (n, n) sparse matrix``, point-wise symmetric positive semi-definite.
    B : (n, m) array or None
        Port matrix; ``None`` for a closed system.
    u : callable or None
        Input ``u(t) -> (m,) array``; ``None`` for a closed system.

    Attributes
    ----------
    E : (n, n) sparse matrix
        Flow matrix.
    rhs : callable
        Right-hand side ``rhs(t, x) -> (n,) array`` in explicit form.
    """

    def __init__(self, E_diag, J, R, B=None, u=None):
        self.E_diag = np.asarray(E_diag, float)
        self.E = sp.diags(self.E_diag, format="csc")
        self.J = J
        self.R = R
        self.B = B
        self.u = u

    def H(self, x):
        """Quadratic Hamiltonian ``H(x) = 0.5 x^T E x``."""
        return 0.5 * x @ (self.E @ x)

    def grad_H(self, x):
        return self.E @ x

    def input_term(self, t):
        """Port term ``B u(t)``; zero for a closed system."""
        return self.B @ self.u(t) if self.B is not None else 0.0

    def rhs(self, t, x):
        return ((self.J - self.R(x)) @ x + self.input_term(t)) / self.E_diag


def setup_lotka_volterra():
    """Lotka-Volterra model of Section 4.1 as a skew-gradient system."""

    def J(x):
        return np.array([[0.0, x[0] * x[1]], [-x[0] * x[1], 0.0]])

    def H(x):
        return x[0] - np.log(x[0]) + x[1] - 2 * np.log(x[1])

    def grad_H(x):
        return np.array([1 - 1 / x[0], 1 - 2 / x[1]])

    return SkewGradientSystem(J, H, grad_H)


def R_hat(eta):
    """Nonlinear resistance of the diode model, evaluated component-wise."""
    eta = np.asarray(eta, dtype=float)
    return np.where(
        eta < 0,
        40e-12,
        40e-12 * (1 + 20 * eta + (40 * eta) ** 2 / 6 + (40 * eta) ** 3 / 24 + (40 * eta) ** 4 / 120),
    )


def current_source(t):
    """Current source of the LC-oscillator chain, periodically extended."""
    period = 0.5 * np.pi * 1e-3
    t_mod = t % period
    return 5 * np.sin(8e3 * t_mod) ** 4 if t_mod <= (np.pi / 8) * 1e-3 else 0.0


def setup_lc_chain(C, L, i_source=None):
    """LC-oscillator chain of Section 4.2 as a port-Hamiltonian ODE.

    Parameters
    ----------
    C, L : (s,) array_like
        Capacitances and inductances of the ``s`` building blocks.
    i_source : callable or None
        Current source ``i_source(t) -> float``; ``None`` yields a closed system.

    Returns
    -------
    pHSystem
        System with state ``x = (e_1, j_1, ..., e_s, j_s)``.
    """
    C = np.asarray(C, float)
    L = np.asarray(L, float)
    s = len(C)
    n = 2 * s

    E_diag = np.column_stack((C, L)).ravel()
    J = sp.block_diag([np.array([[0.0, -1.0], [1.0, 0.0]])] * s, format="csc")

    # Incidence matrix of the diodes: (Lam x)_l = e_l - e_{l+1}
    rows = np.repeat(np.arange(s - 1), 2)
    cols = np.column_stack((2 * np.arange(s - 1), 2 * np.arange(1, s))).ravel()
    vals = np.tile([1.0, -1.0], s - 1)
    Lam = sp.csr_matrix((vals, (rows, cols)), shape=(s - 1, n))
    LamT = Lam.T.tocsr()

    def R(x):
        return (LamT @ sp.diags(R_hat(Lam @ x)) @ Lam).tocsc()

    if i_source is not None:
        B = np.zeros((n, 1))
        B[0, 0] = 1.0
        u = lambda t: np.array([i_source(t)])
    else:
        B = u = None

    return pHSystem(E_diag, J, R, B, u)


def setup_lc_chain_benchmark(s, with_source=True):
    """LC-oscillator chain with the parameters and initial values of Section 4.2.

    Parameters
    ----------
    s : int
        Number of LC-oscillators.
    with_source : bool
        If ``False``, the current source is switched off (dissipative system).

    Returns
    -------
    (pHSystem, (2*s,) array)
        System and initial value.
    """
    ell = np.arange(1, s + 1)
    C = np.where(ell % 2 == 1, 1e-4, 2e-4)
    L = np.where(ell % 2 == 1, 1e-2, 2e-2)
    system = setup_lc_chain(C, L, current_source if with_source else None)
    x0 = np.column_stack((5 * np.ones(s), np.zeros(s))).ravel()
    return system, x0
