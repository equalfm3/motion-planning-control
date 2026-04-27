"""PID controller with anti-windup, derivative filtering, and output saturation.

Implements a discrete-time PID controller using back-calculation anti-windup
to prevent integral windup when actuators saturate. Includes a first-order
low-pass filter on the derivative term to reduce noise amplification.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class PIDGains:
    """PID gain parameters.

    Args:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
    """

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0


@dataclass
class PIDConfig:
    """Full PID controller configuration.

    Args:
        gains: PID gain triplet.
        dt: Sample period in seconds.
        output_min: Minimum actuator output (saturation lower bound).
        output_max: Maximum actuator output (saturation upper bound).
        tracking_tc: Anti-windup tracking time constant. Smaller values
            recover from saturation faster.
        deriv_filter_coeff: Derivative low-pass filter coefficient in [0, 1].
            0 = no filtering, 1 = full filtering (no derivative action).
    """

    gains: PIDGains = field(default_factory=PIDGains)
    dt: float = 0.01
    output_min: float = -float("inf")
    output_max: float = float("inf")
    tracking_tc: float = 0.1
    deriv_filter_coeff: float = 0.1


class PIDController:
    """Discrete-time PID controller with anti-windup and derivative filtering.

    Uses back-calculation anti-windup: when the raw output exceeds actuator
    limits, the difference between saturated and unsaturated output is fed
    back to the integrator, preventing windup.

    Args:
        config: Controller configuration.
    """

    def __init__(self, config: PIDConfig | None = None) -> None:
        self.config = config or PIDConfig()
        self._integral: float = 0.0
        self._prev_error: float = 0.0
        self._prev_derivative: float = 0.0

    def reset(self) -> None:
        """Reset controller internal state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0

    def update(self, error: float) -> float:
        """Compute one PID control step.

        Args:
            error: Current tracking error (setpoint - measurement).

        Returns:
            Saturated control output.
        """
        cfg = self.config
        g = cfg.gains

        # Proportional term
        p_term = g.kp * error

        # Integral term (accumulated before anti-windup correction)
        self._integral += g.ki * error * cfg.dt

        # Derivative term with low-pass filter
        raw_deriv = (error - self._prev_error) / cfg.dt if cfg.dt > 0 else 0.0
        alpha = cfg.deriv_filter_coeff
        filtered_deriv = alpha * self._prev_derivative + (1.0 - alpha) * raw_deriv
        d_term = g.kd * filtered_deriv

        # Raw (unsaturated) output
        u_raw = p_term + self._integral + d_term

        # Saturate output
        u_sat = max(cfg.output_min, min(cfg.output_max, u_raw))

        # Back-calculation anti-windup
        if cfg.tracking_tc > 0:
            self._integral += (u_sat - u_raw) * cfg.dt / cfg.tracking_tc

        # Store state for next iteration
        self._prev_error = error
        self._prev_derivative = filtered_deriv

        return u_sat

    def step_response(
        self, setpoint: float, num_steps: int, plant_fn: callable | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Simulate a step response.

        Args:
            setpoint: Desired output value.
            num_steps: Number of simulation steps.
            plant_fn: Plant dynamics function ``x_next = plant_fn(x, u, dt)``.
                Defaults to a second-order system.

        Returns:
            Tuple of (time, output, control) arrays.
        """
        dt = self.config.dt
        self.reset()

        if plant_fn is None:
            # Default: second-order system  x'' + 2*zeta*wn*x' + wn^2*x = wn^2*u
            wn, zeta = 5.0, 0.7

            def plant_fn(state: NDArray, u: float, dt: float) -> NDArray:
                x, xdot = state
                xddot = wn**2 * (u - x) - 2 * zeta * wn * xdot
                return np.array([x + xdot * dt, xdot + xddot * dt])

        state = np.array([0.0, 0.0])
        times = np.arange(num_steps) * dt
        outputs = np.zeros(num_steps)
        controls = np.zeros(num_steps)

        for i in range(num_steps):
            error = setpoint - state[0]
            u = self.update(error)
            state = plant_fn(state, u, dt)
            outputs[i] = state[0]
            controls[i] = u

        return times, outputs, controls


def _simulate_and_print(kp: float, ki: float, kd: float, setpoint: float) -> None:
    """Run a step response simulation and print summary statistics."""
    config = PIDConfig(
        gains=PIDGains(kp=kp, ki=ki, kd=kd),
        dt=0.01,
        output_min=-10.0,
        output_max=10.0,
    )
    ctrl = PIDController(config)
    times, outputs, controls = ctrl.step_response(setpoint, num_steps=500)

    # Compute metrics
    final_val = outputs[-1]
    ss_error = abs(setpoint - final_val)
    overshoot = (max(outputs) - setpoint) / setpoint * 100 if setpoint != 0 else 0.0
    rise_idx = np.argmax(outputs >= 0.9 * setpoint)
    rise_time = times[rise_idx] if rise_idx > 0 else float("inf")

    print(f"PID Step Response (Kp={kp}, Ki={ki}, Kd={kd}, setpoint={setpoint})")
    print(f"  Final value:     {final_val:.4f}")
    print(f"  Steady-state err:{ss_error:.4f}")
    print(f"  Overshoot:       {overshoot:.1f}%")
    print(f"  Rise time (90%): {rise_time:.3f}s")
    print(f"  Max control:     {max(controls):.2f}")
    print(f"  Min control:     {min(controls):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PID controller step response demo")
    parser.add_argument("--kp", type=float, default=2.0, help="Proportional gain")
    parser.add_argument("--ki", type=float, default=0.5, help="Integral gain")
    parser.add_argument("--kd", type=float, default=0.1, help="Derivative gain")
    parser.add_argument("--setpoint", type=float, default=1.0, help="Target value")
    args = parser.parse_args()

    _simulate_and_print(args.kp, args.ki, args.kd, args.setpoint)
