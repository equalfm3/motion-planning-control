"""Dynamic models for MPC: unicycle, bicycle, and quadrotor.

Each model provides discrete-time dynamics via ``step`` and linearization
via ``linearize`` for use with linear MPC.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class DoubleIntegrator:
    """2D double integrator: state = [x, y, vx, vy], input = [ax, ay]."""

    dt: float = 0.1
    nx: int = 4
    nu: int = 2

    def _matrices(self) -> tuple[NDArray, NDArray]:
        """Return (A, B) discrete-time state-space matrices."""
        dt = self.dt
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        B = np.array([[0.5*dt**2, 0], [0, 0.5*dt**2], [dt, 0], [0, dt]])
        return A, B

    def step(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance state by one timestep."""
        A, B = self._matrices()
        return A @ x + B @ u

    def linearize(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
        """Return (A, B) matrices (system is already linear)."""
        return self._matrices()


@dataclass
class Unicycle:
    """Unicycle model: state = [x, y, theta], input = [v, omega]."""

    dt: float = 0.1
    nx: int = 3
    nu: int = 2

    def step(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance state by one timestep using Euler integration."""
        dt, theta = self.dt, x[2]
        return np.array([
            x[0] + u[0] * math.cos(theta) * dt,
            x[1] + u[0] * math.sin(theta) * dt,
            x[2] + u[1] * dt,
        ])

    def linearize(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
        """Linearize around (x, u) using Jacobians."""
        dt, theta, v = self.dt, x[2], u[0]
        A = np.array([
            [1, 0, -v * math.sin(theta) * dt],
            [0, 1, v * math.cos(theta) * dt],
            [0, 0, 1],
        ])
        B = np.array([
            [math.cos(theta) * dt, 0],
            [math.sin(theta) * dt, 0],
            [0, dt],
        ])
        return A, B


@dataclass
class Bicycle:
    """Kinematic bicycle model: state = [x, y, theta, v], input = [a, delta].

    Args:
        dt: Discretization timestep.
        wheelbase: Distance between front and rear axles.
    """

    dt: float = 0.1
    wheelbase: float = 2.5
    nx: int = 4
    nu: int = 2

    def step(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance state by one timestep."""
        dt, L = self.dt, self.wheelbase
        px, py, theta, v = x
        a, delta = u
        return np.array([
            px + v * math.cos(theta) * dt,
            py + v * math.sin(theta) * dt,
            theta + (v / L) * math.tan(delta) * dt,
            v + a * dt,
        ])

    def linearize(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
        """Linearize around (x, u)."""
        dt, L = self.dt, self.wheelbase
        theta, v, delta = x[2], x[3], u[1]
        A = np.array([
            [1, 0, -v * math.sin(theta) * dt, math.cos(theta) * dt],
            [0, 1, v * math.cos(theta) * dt, math.sin(theta) * dt],
            [0, 0, 1, math.tan(delta) / L * dt],
            [0, 0, 0, 1],
        ])
        B = np.array([
            [0, 0], [0, 0],
            [0, v / (L * math.cos(delta) ** 2) * dt],
            [dt, 0],
        ])
        return A, B


@dataclass
class Quadrotor2D:
    """Planar quadrotor: state = [x, z, theta, vx, vz, omega], input = [f1, f2].

    Args:
        dt: Discretization timestep.
        mass: Quadrotor mass in kg.
        inertia: Moment of inertia.
        arm_length: Distance from center to rotor.
        gravity: Gravitational acceleration.
    """

    dt: float = 0.05
    mass: float = 1.0
    inertia: float = 0.01
    arm_length: float = 0.2
    gravity: float = 9.81
    nx: int = 6
    nu: int = 2

    def step(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Advance state by one timestep."""
        dt = self.dt
        m, I, l, g = self.mass, self.inertia, self.arm_length, self.gravity
        px, pz, theta, vx, vz, omega = x
        f_total, tau = u[0] + u[1], l * (u[1] - u[0])
        ax = -f_total * math.sin(theta) / m
        az = f_total * math.cos(theta) / m - g
        return np.array([
            px + vx * dt, pz + vz * dt, theta + omega * dt,
            vx + ax * dt, vz + az * dt, omega + tau / I * dt,
        ])

    def linearize(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
        """Linearize around (x, u)."""
        dt, m, I, l = self.dt, self.mass, self.inertia, self.arm_length
        theta, f_total = x[2], u[0] + u[1]
        A = np.eye(6)
        A[0, 3] = A[1, 4] = A[2, 5] = dt
        A[3, 2] = -f_total * math.cos(theta) / m * dt
        A[4, 2] = -f_total * math.sin(theta) / m * dt
        B = np.zeros((6, 2))
        s, c = math.sin(theta), math.cos(theta)
        B[3, 0] = B[3, 1] = -s / m * dt
        B[4, 0] = B[4, 1] = c / m * dt
        B[5, 0], B[5, 1] = -l / I * dt, l / I * dt
        return A, B


def get_model(model_type: str, dt: float = 0.1) -> DoubleIntegrator | Unicycle | Bicycle | Quadrotor2D:
    """Factory function to create a model by name.

    Args:
        model_type: One of ``"double_integrator"``, ``"unicycle"``,
            ``"bicycle"``, ``"quadrotor_2d"``.
        dt: Discretization timestep.

    Returns:
        Instantiated model.
    """
    models = {
        "double_integrator": lambda: DoubleIntegrator(dt=dt),
        "unicycle": lambda: Unicycle(dt=dt),
        "bicycle": lambda: Bicycle(dt=dt),
        "quadrotor_2d": lambda: Quadrotor2D(dt=dt),
    }
    if model_type not in models:
        raise ValueError(f"Unknown model '{model_type}', choose from {list(models.keys())}")
    return models[model_type]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic models demo")
    parser.add_argument(
        "--model",
        choices=["double_integrator", "unicycle", "bicycle", "quadrotor_2d"],
        default="double_integrator",
    )
    parser.add_argument("--steps", type=int, default=50, help="Simulation steps")
    args = parser.parse_args()

    model = get_model(args.model)
    x = np.zeros(model.nx)
    u = np.ones(model.nu) * 0.5

    print(f"Model: {args.model} (nx={model.nx}, nu={model.nu})")
    print(f"Constant input: {u}\n")
    for i in range(args.steps):
        x = model.step(x, u)
        if i % 10 == 0 or i == args.steps - 1:
            print(f"  Step {i:3d}: [{', '.join(f'{v:.3f}' for v in x)}]")
