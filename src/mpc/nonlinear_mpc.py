"""Nonlinear Model Predictive Control via sequential linearization.

Implements nonlinear MPC by iteratively linearizing the dynamics around
the current trajectory and solving a linear QP at each iteration (iLQR-style).
This avoids the CasADi/IPOPT dependency while still handling nonlinear
dynamics like the unicycle and bicycle models.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import Unicycle, Bicycle, get_model


@dataclass
class NMPCConfig:
    """Configuration for nonlinear MPC.

    Args:
        horizon: Prediction horizon.
        Q: State cost weight matrix.
        R: Input cost weight matrix.
        Qf: Terminal cost weight matrix.
        u_min: Input lower bounds.
        u_max: Input upper bounds.
        sqp_iters: Number of sequential QP iterations per solve.
        line_search_beta: Line search backtracking factor.
    """

    horizon: int = 15
    Q: NDArray[np.float64] | None = None
    R: NDArray[np.float64] | None = None
    Qf: NDArray[np.float64] | None = None
    u_min: NDArray[np.float64] | None = None
    u_max: NDArray[np.float64] | None = None
    sqp_iters: int = 3
    line_search_beta: float = 0.5


class NonlinearMPC:
    """Nonlinear MPC using sequential linearization (SQP-style).

    At each control step, the dynamics are linearized around the current
    nominal trajectory, a QP is solved for the correction, and the
    trajectory is updated. Multiple SQP iterations refine the solution.

    Args:
        model: Dynamic model with ``step`` and ``linearize`` methods.
        config: NMPC configuration.
    """

    def __init__(self, model: object, config: NMPCConfig | None = None) -> None:
        self.model = model
        self.nx: int = model.nx
        self.nu: int = model.nu
        self.config = config or NMPCConfig()
        self._setup_defaults()
        self._nominal_u: NDArray[np.float64] | None = None

    def _setup_defaults(self) -> None:
        """Fill in default matrices."""
        cfg = self.config
        if cfg.Q is None:
            cfg.Q = np.eye(self.nx)
        if cfg.R is None:
            cfg.R = np.eye(self.nu) * 0.1
        if cfg.Qf is None:
            cfg.Qf = cfg.Q * 10.0
        if cfg.u_min is None:
            cfg.u_min = np.full(self.nu, -float("inf"))
        if cfg.u_max is None:
            cfg.u_max = np.full(self.nu, float("inf"))

    def _rollout(
        self, x0: NDArray[np.float64], U: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Forward-simulate the nonlinear dynamics.

        Args:
            x0: Initial state.
            U: Input sequence of shape (N, nu).

        Returns:
            State trajectory of shape (N+1, nx).
        """
        N = len(U)
        X = np.zeros((N + 1, self.nx))
        X[0] = x0
        for k in range(N):
            X[k + 1] = self.model.step(X[k], U[k])
        return X

    def _cost(
        self,
        X: NDArray[np.float64],
        U: NDArray[np.float64],
        x_ref: NDArray[np.float64],
    ) -> float:
        """Compute total trajectory cost.

        Args:
            X: State trajectory (N+1, nx).
            U: Input sequence (N, nu).
            x_ref: Reference state.

        Returns:
            Scalar cost value.
        """
        cfg = self.config
        cost = 0.0
        N = len(U)
        for k in range(N):
            dx = X[k] - x_ref
            cost += dx @ cfg.Q @ dx + U[k] @ cfg.R @ U[k]
        dx_f = X[N] - x_ref
        cost += dx_f @ cfg.Qf @ dx_f
        return cost

    def solve(
        self,
        x0: NDArray[np.float64],
        x_ref: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve the NMPC problem via sequential linearization.

        Args:
            x0: Current state.
            x_ref: Reference state (defaults to origin).

        Returns:
            Tuple of (optimal_inputs, predicted_states).
        """
        cfg = self.config
        N = cfg.horizon

        if x_ref is None:
            x_ref = np.zeros(self.nx)

        # Warm start: shift previous solution or initialize to zero
        if self._nominal_u is not None and len(self._nominal_u) == N:
            U = np.roll(self._nominal_u, -1, axis=0)
            U[-1] = np.zeros(self.nu)
        else:
            U = np.zeros((N, self.nu))

        U = np.clip(U, cfg.u_min, cfg.u_max)

        for sqp_iter in range(cfg.sqp_iters):
            X = self._rollout(x0, U)

            # Backward pass: compute gains via linearized LQR
            # Value function: V_k = x^T P_k x + 2 p_k^T x
            P = cfg.Qf.copy()
            p = cfg.Qf @ (X[N] - x_ref)

            K_gains = np.zeros((N, self.nu, self.nx))
            k_ff = np.zeros((N, self.nu))

            for k in range(N - 1, -1, -1):
                A_k, B_k = self.model.linearize(X[k], U[k])
                dx = X[k] - x_ref

                # Q-function matrices
                Qxx = cfg.Q + A_k.T @ P @ A_k
                Quu = cfg.R + B_k.T @ P @ B_k
                Qux = B_k.T @ P @ A_k
                qx = cfg.Q @ dx + A_k.T @ p
                qu = cfg.R @ U[k] + B_k.T @ p

                # Regularize for numerical stability
                Quu_reg = Quu + np.eye(self.nu) * 1e-6
                Quu_inv = np.linalg.solve(Quu_reg, np.eye(self.nu))

                K_gains[k] = -Quu_inv @ Qux
                k_ff[k] = -Quu_inv @ qu

                P = Qxx + Qux.T @ K_gains[k]
                p = qx + Qux.T @ k_ff[k]

            # Forward pass with line search
            alpha = 1.0
            current_cost = self._cost(X, U, x_ref)

            for _ in range(10):
                U_new = np.zeros_like(U)
                X_new = np.zeros_like(X)
                X_new[0] = x0

                for k in range(N):
                    dx = X_new[k] - X[k]
                    U_new[k] = U[k] + K_gains[k] @ dx + alpha * k_ff[k]
                    U_new[k] = np.clip(U_new[k], cfg.u_min, cfg.u_max)
                    X_new[k + 1] = self.model.step(X_new[k], U_new[k])

                new_cost = self._cost(X_new, U_new, x_ref)
                if new_cost < current_cost:
                    U = U_new
                    break
                alpha *= cfg.line_search_beta

        self._nominal_u = U.copy()
        X_final = self._rollout(x0, U)
        return U, X_final

    def control(
        self, x0: NDArray[np.float64], x_ref: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Get the first optimal control action.

        Args:
            x0: Current state.
            x_ref: Reference state.

        Returns:
            Optimal control for the current timestep.
        """
        U_opt, _ = self.solve(x0, x_ref)
        return U_opt[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nonlinear MPC demo")
    parser.add_argument(
        "--model", choices=["unicycle", "bicycle"], default="unicycle"
    )
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    model = get_model(args.model, dt=0.1)
    nmpc_cfg = NMPCConfig(
        horizon=args.horizon,
        Q=np.diag([10.0, 10.0, 0.1][:model.nx]),
        R=np.eye(model.nu) * 0.1,
        u_min=np.full(model.nu, -2.0),
        u_max=np.full(model.nu, 2.0),
    )
    nmpc = NonlinearMPC(model, nmpc_cfg)

    x = np.zeros(model.nx)
    x_ref = np.zeros(model.nx)
    x_ref[0] = 5.0
    x_ref[1] = 3.0

    print(f"Nonlinear MPC ({args.model}, horizon={args.horizon})")
    print(f"  Start: [{', '.join(f'{v:.2f}' for v in x)}]")
    print(f"  Goal:  [{', '.join(f'{v:.2f}' for v in x_ref)}]")
    print()

    for i in range(args.steps):
        u = nmpc.control(x, x_ref)
        x = model.step(x, u)
        if i % 10 == 0 or i == args.steps - 1:
            err = np.linalg.norm(x[:2] - x_ref[:2])
            print(f"  Step {i:3d}: pos=({x[0]:.2f}, {x[1]:.2f}) err={err:.3f} u=[{', '.join(f'{v:.2f}' for v in u)}]")
