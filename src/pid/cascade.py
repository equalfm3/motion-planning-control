"""Cascaded PID control: nested position → velocity → torque loops.

In real robotic systems, control loops are nested at different frequencies.
The outer loop (position) runs slowest and commands a velocity setpoint.
The middle loop (velocity) commands a torque setpoint. The inner loop
(torque/current) runs fastest and drives the actuator.

This module implements a generic cascaded PID architecture with configurable
loop rates and inter-loop synchronization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .controller import PIDConfig, PIDController, PIDGains


@dataclass
class LoopConfig:
    """Configuration for one loop in the cascade.

    Args:
        name: Human-readable loop name.
        pid_config: PID configuration for this loop.
        rate_hz: Loop execution frequency in Hz.
    """

    name: str = "loop"
    pid_config: PIDConfig = field(default_factory=PIDConfig)
    rate_hz: float = 100.0


@dataclass
class CascadeConfig:
    """Configuration for the full cascade.

    Args:
        outer: Outermost (slowest) loop — typically position.
        middle: Middle loop — typically velocity.
        inner: Innermost (fastest) loop — typically torque.
    """

    outer: LoopConfig = field(default_factory=lambda: LoopConfig(
        name="position",
        pid_config=PIDConfig(
            gains=PIDGains(kp=4.0, ki=0.2, kd=0.5),
            dt=0.01,
            output_min=-5.0,
            output_max=5.0,
        ),
        rate_hz=100.0,
    ))
    middle: LoopConfig = field(default_factory=lambda: LoopConfig(
        name="velocity",
        pid_config=PIDConfig(
            gains=PIDGains(kp=2.0, ki=0.5, kd=0.01),
            dt=0.001,
            output_min=-10.0,
            output_max=10.0,
        ),
        rate_hz=1000.0,
    ))
    inner: LoopConfig = field(default_factory=lambda: LoopConfig(
        name="torque",
        pid_config=PIDConfig(
            gains=PIDGains(kp=10.0, ki=1.0, kd=0.0),
            dt=0.0001,
            output_min=-20.0,
            output_max=20.0,
        ),
        rate_hz=10000.0,
    ))


class CascadedPID:
    """Three-level cascaded PID controller.

    The outer loop produces a setpoint for the middle loop, which produces
    a setpoint for the inner loop. Each loop runs at its own rate.

    Args:
        config: Cascade configuration.
    """

    def __init__(self, config: CascadeConfig | None = None) -> None:
        self.config = config or CascadeConfig()
        self.outer = PIDController(self.config.outer.pid_config)
        self.middle = PIDController(self.config.middle.pid_config)
        self.inner = PIDController(self.config.inner.pid_config)

    def reset(self) -> None:
        """Reset all three loops."""
        self.outer.reset()
        self.middle.reset()
        self.inner.reset()

    def update(
        self,
        pos_ref: float,
        pos_meas: float,
        vel_meas: float,
        torque_meas: float,
    ) -> float:
        """Compute one cascade update at the outer loop rate.

        Runs the middle and inner loops at their respective sub-rates
        within a single outer loop period.

        Args:
            pos_ref: Desired position.
            pos_meas: Measured position.
            vel_meas: Measured velocity.
            torque_meas: Measured torque (or current).

        Returns:
            Final actuator command from the inner loop.
        """
        # Outer loop: position → velocity setpoint
        pos_error = pos_ref - pos_meas
        vel_setpoint = self.outer.update(pos_error)

        # Middle loop: velocity → torque setpoint
        vel_error = vel_setpoint - vel_meas
        torque_setpoint = self.middle.update(vel_error)

        # Inner loop: torque → actuator command
        torque_error = torque_setpoint - torque_meas
        actuator_cmd = self.inner.update(torque_error)

        return actuator_cmd


def simulate_cascade(
    cascade: CascadedPID,
    pos_ref: float,
    duration: float,
    dt: float = 0.01,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Simulate cascaded PID on a simple inertial plant.

    Plant model: mass-spring-damper with J*theta'' + b*theta' = tau.

    Args:
        cascade: Cascaded PID controller.
        pos_ref: Desired position.
        duration: Simulation duration in seconds.
        dt: Simulation timestep.

    Returns:
        Tuple of (time, position, velocity, control) arrays.
    """
    num_steps = int(duration / dt)
    J = 1.0   # inertia
    b = 0.5   # damping

    pos = 0.0
    vel = 0.0
    torque_actual = 0.0

    times = np.arange(num_steps) * dt
    positions = np.zeros(num_steps)
    velocities = np.zeros(num_steps)
    controls = np.zeros(num_steps)

    cascade.reset()

    for i in range(num_steps):
        u = cascade.update(pos_ref, pos, vel, torque_actual)

        # Plant dynamics
        acc = (u - b * vel) / J
        vel += acc * dt
        pos += vel * dt
        torque_actual = u  # assume direct torque application

        positions[i] = pos
        velocities[i] = vel
        controls[i] = u

    return times, positions, velocities, controls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cascaded PID demo")
    parser.add_argument("--setpoint", type=float, default=1.0, help="Position setpoint")
    parser.add_argument("--duration", type=float, default=3.0, help="Simulation duration (s)")
    args = parser.parse_args()

    cascade = CascadedPID()
    times, pos, vel, ctrl = simulate_cascade(cascade, args.setpoint, args.duration)

    ss_error = abs(args.setpoint - pos[-1])
    overshoot = (max(pos) - args.setpoint) / args.setpoint * 100 if args.setpoint != 0 else 0.0

    print(f"Cascaded PID (position → velocity → torque)")
    print(f"  Setpoint:        {args.setpoint}")
    print(f"  Final position:  {pos[-1]:.4f}")
    print(f"  Steady-state err:{ss_error:.4f}")
    print(f"  Overshoot:       {overshoot:.1f}%")
    print(f"  Max velocity:    {max(vel):.3f}")
    print(f"  Max control:     {max(ctrl):.2f}")
