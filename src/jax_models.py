"""
Controller architecture. PsiU is a JAX/Equinox port of acyclic REN (Recurrent Equilibrium
Network) with Lyapunov-stable parameter constraints. PsiX wraps a nominal one-step predictor
for the internal model. Controller assembles both, plus an optional input schedule.
"""

from __future__ import annotations

from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

Omega = tuple[jax.Array, jax.Array]


class PsiU(eqx.Module):
    """Acyclic REN implementation."""

    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    n_xi: int = eqx.field(static=True)
    l: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    inner_output_gain: float = eqx.field(static=True)

    X: jax.Array
    Y: jax.Array
    B2: jax.Array
    C2: jax.Array
    D21: jax.Array
    D22: jax.Array
    D12: jax.Array

    def __init__(
        self,
        n: int,
        m: int,
        n_xi: int,
        l: int,
        *,
        key: jax.Array,
        std_ini_param: float = 0.1,
        epsilon: float = 1e-3,
        inner_output_gain: float = 20.0,
    ) -> None:
        self.n = n
        self.m = m
        self.n_xi = n_xi
        self.l = l
        self.epsilon = epsilon
        self.inner_output_gain = inner_output_gain

        k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)
        std = std_ini_param
        self.X = std * jr.normal(k1, (2 * n_xi + l, 2 * n_xi + l))
        self.Y = std * jr.normal(k2, (n_xi, n_xi))
        self.B2 = std * jr.normal(k3, (n_xi, n))
        self.C2 = std * jr.normal(k4, (m, n_xi))
        self.D21 = std * jr.normal(k5, (m, l))
        self.D22 = std * jr.normal(k6, (m, n))
        self.D12 = std * jr.normal(k7, (l, n))

    def _derived_matrices(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Build constrained REN matrices from free parameters."""
        n_xi = self.n_xi
        l = self.l
        dtype = self.X.dtype

        H = self.X.T @ self.X + self.epsilon * jnp.eye(2 * n_xi + l, dtype=dtype)

        H11 = H[:n_xi, :n_xi]
        H21 = H[n_xi : n_xi + l, :n_xi]
        H22 = H[n_xi : n_xi + l, n_xi : n_xi + l]
        H31 = H[n_xi + l :, :n_xi]
        H32 = H[n_xi + l :, n_xi : n_xi + l]
        H33 = H[n_xi + l :, n_xi + l :]

        P = H33
        F = H31
        B1 = H32
        E = 0.5 * (H11 + P + self.Y - self.Y.T)
        Lambda = 0.5 * jnp.diag(H22)
        D11 = -jnp.tril(H22, k=-1)
        C1 = -H21
        return F, B1, E, Lambda, C1, D11

    def __call__(self, t: int | jax.Array, w: jax.Array, xi: jax.Array) -> tuple[jax.Array, jax.Array]:
        del t
        F, B1, E, Lambda, C1, D11 = self._derived_matrices()

        def body(i: int, epsilon_vec: jax.Array) -> jax.Array:
            v = C1[i] @ xi + D11[i] @ epsilon_vec + self.D12[i] @ w
            eps_i = jnp.tanh(v / Lambda[i])
            return epsilon_vec.at[i].set(eps_i)

        epsilon_vec = jax.lax.fori_loop(
            0,
            self.l,
            body,
            jnp.zeros((self.l,), dtype=w.dtype),
        )

        E_xi_next = F @ xi + B1 @ epsilon_vec + self.B2 @ w
        xi_next = jnp.linalg.solve(E, E_xi_next)
        u = self.C2 @ xi + self.D21 @ epsilon_vec + self.D22 @ w
        return self.inner_output_gain * u, xi_next


class PsiX(eqx.Module):
    """Wrapper around a nominal one-step predictor f(t, y, u)."""
    f: Callable[[int | jax.Array, jax.Array, jax.Array], jax.Array] = eqx.field(static=True)

    def __init__(self, f: Callable[[int | jax.Array, jax.Array, jax.Array], jax.Array]) -> None:
        self.f = f

    def __call__(self, t: int | jax.Array, omega: Omega) -> tuple[jax.Array, None]:
        y, u = omega
        psi_x = self.f(t, y, u)
        return psi_x, None


class InputSchedule(eqx.Module):
    """Port of Input; use with vmap/scan outside the module."""

    m: int = eqx.field(static=True)
    t_end: int = eqx.field(static=True)
    u: jax.Array

    def __init__(
        self,
        m: int,
        t_end: int,
        *,
        active: bool = True,
        key: Optional[jax.Array] = None,
        std: float = 0.0,
    ) -> None:
        self.m = m
        self.t_end = t_end
        if active:
            if key is None:
                raise ValueError("A PRNG key is required when active=True.")
            self.u = std * jr.normal(key, (t_end, m))
        else:
            self.u = jnp.zeros((t_end, m))

    def __call__(self, t: int | jax.Array) -> jax.Array:
        t_int = jnp.asarray(t, dtype=jnp.int32)
        idx = jnp.clip(t_int, 0, max(self.t_end - 1, 0))
        value = self.u[idx]
        return jax.lax.cond(
            t_int < self.t_end,
            lambda _: value,
            lambda _: jnp.zeros((self.m,), dtype=self.u.dtype),
            operand=None,
        )


class Controller(eqx.Module):
    """Acyclic REN controller used by the MJX rollout."""

    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)
    use_sp: bool = eqx.field(static=True)
    output_amplification: float = eqx.field(static=True)

    psi_x: PsiX
    psi_u: PsiU
    sp: Optional[InputSchedule]

    def __init__(
        self,
        f: Callable[[int | jax.Array, jax.Array, jax.Array], jax.Array],
        n: int,
        m: int,
        n_xi: int,
        l: int,
        *,
        key: jax.Array,
        use_sp: bool = False,
        t_end_sp: Optional[int] = None,
        std_ini_param: float = 0.1,
        output_amplification: float = 20.0,
        psi_u_inner_output_gain: float = 20.0,
    ) -> None:
        key_psi_u, key_sp = jr.split(key, 2)
        self.n = n
        self.m = m
        self.use_sp = use_sp
        self.output_amplification = output_amplification
        self.psi_x = PsiX(f)
        self.psi_u = PsiU(
            n,
            m,
            n_xi,
            l,
            key=key_psi_u,
            std_ini_param=std_ini_param,
            inner_output_gain=psi_u_inner_output_gain,
        )
        self.sp = (
            InputSchedule(n, int(t_end_sp), active=True, key=key_sp)
            if use_sp and t_end_sp is not None
            else None
        )

    def step_from_omega(
        self,
        t: int | jax.Array,
        y: jax.Array,
        xi: jax.Array,
        omega: Omega,
    ) -> tuple[jax.Array, jax.Array, Omega]:
        """Step the controller from the previous omega pair."""
        f_hat, _ = self.psi_x(t, omega)
        return self.step_from_prediction(t, y, xi, f_hat)

    def step_from_signal(
        self,
        t: int | jax.Array,
        w_hat: jax.Array,
        xi: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the REN on an already-formed feedback signal.
        """
        if self.use_sp and self.sp is not None:
            w_hat = w_hat + self.sp(t)
        u, xi_next = self.psi_u(t, w_hat, xi)
        return u * self.output_amplification, xi_next

    def step_from_prediction(
        self,
        t: int | jax.Array,
        y: jax.Array,
        xi: jax.Array,
        f_hat: jax.Array,
    ) -> tuple[jax.Array, jax.Array, Omega]:
        """Step the controller from an externally supplied prediction."""
        w_hat = y - f_hat
        u, xi_next = self.step_from_signal(t, w_hat, xi)
        omega_next = (y, u)
        return u, xi_next, omega_next

    def __call__(
        self,
        t: int | jax.Array,
        y: jax.Array,
        xi: jax.Array,
        omega: Omega,
    ) -> tuple[jax.Array, jax.Array, Omega]:
        return self.step_from_omega(t, y, xi, omega)
