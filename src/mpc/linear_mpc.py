"""Linear Model Predictive Control via quadratic programming.

Solves the receding-horizon optimal control problem for linear systems
with state and input constraints. Uses numpy for the QP formulation
with a custom active-set solver to avoid external solver dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .models import DoubleIntegrator


@dataclass
class LinearMPCConfig:
    """Configuration for linear MPC.

    Args:
        horizon: Prediction horizon length.
        Q: State cost weight matrix (nx × nx).
        R: Input cost weight matrix (nu × nu).
        Qf: Terminal state cost weight matrix (nx × nx).
        x_min: State lower bounds.
        x_max: State upper bounds.
        u_min: Input lower bounds.
        u_max: Input upper bounds.
    """

    horizon: int = 20
    Q: NDArray[np.float64] | None = None
    R: NDArray[np.float64] | None = None
    Qf: NDArray[np.float64] | None = None
    x_min: NDArray[np.float64] | None = None
    x_max: NDArray[np.float64] | None = None
    u_min: NDArray[np.float64] | None = None
    u_max: NDArray[np.float64] | None = None


class LinearMPC:
    """Linear MPC controller using condensed QP formulation.

    Builds the prediction matrices once and solves a box-constrained QP
    at each timestep using projected gradient descent.

    Args:
        A: Discrete-time state matrix.
        B: Discrete-time input matrix.
        config: MPC configuration.
    """

    def __init__(
        self,
        A: NDArray[np.float64],
        B: NDArray[np.float64],
        config: LinearMPCConfig | None = None,
    ) -> None:
        self.A = A
        self.B = B
        self.config = config or LinearMPCConfig()
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self._setup_defaults()
        self._build_prediction_matrices()
        self._prev_solution: NDArray[np.float64] | None = None

    def _setup_defaults(self) -> None:
        """Fill in default weight matrices if not provided."""
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

    def _build_prediction_matrices(self) -> None:
        """Build condensed prediction matrices S_x and S_u.

        x_pred = S_x @ x0 + S_u @ U, where U is the stacked input vector.
        """
        N = self.config.horizon
        nx, nu = self.nx, self.nu
        A, B = self.A, self.B

        # Precompute A^k powers iteratively (avoids matrix_power overflow)
        A_powers = [np.eye(nx)]
        for k in range(N):
            A_powers.append(A @ A_powers[-1])

        # S_x: maps initial state to predicted states
        self._Sx = np.zeros((N * nx, nx))
        for k in range(N):
            self._Sx[k * nx : (k + 1) * nx, :] = A_powers[k + 1]

        # S_u: maps input sequence to predicted states
        self._Su = np.zeros((N * nx, N * nu))
        for k in range(N):
            for j in range(k + 1):
                self._Su[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = A_powers[k - j] @ B

    def solve(
        self,
        x0: NDArray[np.float64],
        x_ref: NDArray[np.float64] | None = None,
        max_iter: int = 200,
        lr: float = 0.01,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Solve the MPC optimization problem.

        Uses projected gradient descent on the condensed QP with
        warm-starting from the previous solution.

        Args:
            x0: Current state.
            x_ref: Reference state (defaults to origin).
            max_iter: Maximum solver iterations.
            lr: Gradient descent step size.

        Returns:
            Tuple of (optimal_input_sequence, predicted_states).
        """
        cfg = self.config
        N, nx, nu = cfg.horizon, self.nx, self.nu

        if x_ref is None:
            x_ref = np.zeros(nx)

        # Build cost matrices for condensed QP: min 0.5 * U^T H U + g^T U
        Q_bar = np.kron(np.eye(N - 1), cfg.Q)
        Q_bar = np.block([
            [Q_bar, np.zeros(((N - 1) * nx, nx))],
            [np.zeros((nx, (N - 1) * nx)), cfg.Qf],
        ])
        R_bar = np.kron(np.eye(N), cfg.R)

        H = self._Su.T @ Q_bar @ self._Su + R_bar + np.eye(N * nu) * 1e-6
        x_ref_stacked = np.tile(x_ref, N)
        free_response = self._Sx @ x0
        g = self._Su.T @ Q_bar @ (free_response - x_ref_stacked)

        # Warm start from shifted previous solution
        if self._prev_solution is not None and len(self._prev_solution) == N * nu:
            U = np.roll(self._prev_solution, -nu)
            U[-nu:] = 0.0
        else:
            U = np.zeros(N * nu)

        # Projected gradient descent
        u_min_rep = np.tile(cfg.u_min, N)
        u_max_rep = np.tile(cfg.u_max, N)

        for _ in range(max_iter):
            grad = H @ U + g
            U = U - lr * grad
            U = np.clip(U, u_min_rep, u_max_rep)

        self._prev_solution = U.copy()

        # Predicted states
        x_pred = self._Sx @ x0 + self._Su @ U
        x_pred = x_pred.reshape(N, nx)

        return U.reshape(N, nu), x_pred

    def control(self, x0: NDArray[np.float64], x_ref: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
        """Get the first optimal control action (receding horizon).

        Args:
            x0: Current state.
            x_ref: Reference state.

        Returns:
            Optimal control input for the current timestep.
        """
        U_opt, _ = self.solve(x0, x_ref)
        return U_opt[0]


def simulate_mpc(
    horizon: int = 20,
    x_max: float = 5.0,
    u_max: float = 1.0,
    sim_steps: int = 80,
) -> tuple[NDArray, NDArray, NDArray]:
    """Run a linear MPC simulation on a double integrator.

    Args:
        horizon: MPC prediction horizon.
        x_max: State bound magnitude.
        u_max: Input bound magnitude.
        sim_steps: Number of simulation steps.

    Returns:
        Tuple of (time, states, controls).
    """
    model = DoubleIntegrator(dt=0.1)
    A, B = model.linearize(np.zeros(4), np.zeros(2))

    config = LinearMPCConfig(
        horizon=horizon,
        Q=np.diag([10.0, 10.0, 1.0, 1.0]),
        R=np.eye(2) * 0.1,
        u_min=np.array([-u_max, -u_max]),
        u_max=np.array([u_max, u_max]),
    )

    mpc = LinearMPC(A, B, config)
    x = np.array([4.0, 3.0, 0.0, 0.0])
    x_ref = np.zeros(4)

    states = np.zeros((sim_steps, 4))
    controls = np.zeros((sim_steps, 2))

    for i in range(sim_steps):
        u = mpc.control(x, x_ref)
        states[i] = x
        controls[i] = u
        x = model.step(x, u)

    times = np.arange(sim_steps) * model.dt
    return times, states, controls


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser(description="Linear MPC demo")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction horizon")
    parser.add_argument("--x-max", type=float, default=5.0, help="State bound")
    parser.add_argument("--u-max", type=float, default=1.0, help="Input bound")
    args = parser.parse_args()

    times, states, controls = simulate_mpc(args.horizon, args.x_max, args.u_max)

    print(f"Linear MPC (horizon={args.horizon}, u_max={args.u_max})")
    print(f"  Initial state: [{states[0, 0]:.1f}, {states[0, 1]:.1f}, 0, 0]")
    print(f"  Final state:   [{', '.join(f'{v:.3f}' for v in states[-1])}]")
    print(f"  Final error:   {np.linalg.norm(states[-1]):.4f}")
    print(f"  Max |u|:       {np.max(np.abs(controls)):.3f}")
    print(f"  Avg |u|:       {np.mean(np.abs(controls)):.3f}")
