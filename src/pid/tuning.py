"""PID auto-tuning via Ziegler-Nichols and relay feedback methods.

Provides two classical tuning approaches:
- Ziegler-Nichols: Determines gains from the ultimate gain and period found
  by increasing proportional gain until sustained oscillation.
- Relay feedback: Uses a relay (bang-bang) controller to induce oscillation,
  then extracts the ultimate gain and period from the response.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .controller import PIDConfig, PIDController, PIDGains


@dataclass
class TuningResult:
    """Result of a PID auto-tuning procedure.

    Args:
        gains: Computed PID gains.
        ultimate_gain: Critical gain at sustained oscillation.
        ultimate_period: Oscillation period at critical gain.
        method: Tuning method used.
    """

    gains: PIDGains
    ultimate_gain: float
    ultimate_period: float
    method: str


def ziegler_nichols_table(ku: float, tu: float, mode: str = "pid") -> PIDGains:
    """Compute PID gains from Ziegler-Nichols tuning rules.

    Args:
        ku: Ultimate gain (gain at sustained oscillation).
        tu: Ultimate period in seconds.
        mode: Controller type — ``"p"``, ``"pi"``, or ``"pid"``.

    Returns:
        Computed PID gains.

    Raises:
        ValueError: If mode is not one of ``"p"``, ``"pi"``, ``"pid"``.
    """
    if mode == "p":
        return PIDGains(kp=0.5 * ku)
    elif mode == "pi":
        return PIDGains(kp=0.45 * ku, ki=0.54 * ku / tu)
    elif mode == "pid":
        return PIDGains(kp=0.6 * ku, ki=1.2 * ku / tu, kd=0.075 * ku * tu)
    else:
        raise ValueError(f"Unknown mode '{mode}', expected 'p', 'pi', or 'pid'")


def find_ultimate_gain(
    plant_fn: callable,
    dt: float = 0.01,
    k_start: float = 0.1,
    k_step: float = 0.1,
    k_max: float = 100.0,
    sim_steps: int = 2000,
) -> tuple[float, float]:
    """Find the ultimate gain and period by increasing proportional gain.

    Simulates the plant with pure proportional control, increasing the gain
    until the output oscillates with constant amplitude (neither growing
    nor decaying).

    Args:
        plant_fn: Plant dynamics ``state_next = plant_fn(state, u, dt)``.
        dt: Simulation timestep.
        k_start: Starting proportional gain.
        k_step: Gain increment per trial.
        k_max: Maximum gain to try.
        sim_steps: Steps per simulation trial.

    Returns:
        Tuple of (ultimate_gain, ultimate_period).
    """
    kp = k_start
    best_ku = k_start
    best_tu = 1.0

    while kp <= k_max:
        config = PIDConfig(gains=PIDGains(kp=kp), dt=dt)
        ctrl = PIDController(config)
        state = np.array([0.0, 0.0])
        outputs = np.zeros(sim_steps)

        for i in range(sim_steps):
            error = 1.0 - state[0]
            u = ctrl.update(error)
            state = plant_fn(state, u, dt)
            outputs[i] = state[0]

        # Check last quarter for sustained oscillation
        tail = outputs[sim_steps * 3 // 4 :]
        amplitude = (np.max(tail) - np.min(tail)) / 2.0

        if amplitude > 0.01:
            # Estimate period from zero crossings
            centered = tail - np.mean(tail)
            crossings = np.where(np.diff(np.sign(centered)))[0]
            if len(crossings) >= 2:
                avg_half_period = np.mean(np.diff(crossings)) * dt
                best_ku = kp
                best_tu = 2.0 * avg_half_period
                break

        kp += k_step

    return best_ku, best_tu


def relay_feedback_tuning(
    plant_fn: callable,
    relay_amplitude: float = 1.0,
    dt: float = 0.01,
    sim_steps: int = 3000,
    setpoint: float = 0.5,
) -> TuningResult:
    """Tune PID gains using the relay feedback method.

    Applies a relay (bang-bang) controller to induce limit-cycle oscillation,
    then extracts the ultimate gain and period from the response.

    Args:
        plant_fn: Plant dynamics ``state_next = plant_fn(state, u, dt)``.
        relay_amplitude: Relay switching amplitude.
        dt: Simulation timestep.
        sim_steps: Number of simulation steps.
        setpoint: Reference value for the relay.

    Returns:
        Tuning result with computed gains.
    """
    state = np.array([0.0, 0.0])
    outputs = np.zeros(sim_steps)

    for i in range(sim_steps):
        error = setpoint - state[0]
        u = relay_amplitude if error > 0 else -relay_amplitude
        state = plant_fn(state, u, dt)
        outputs[i] = state[0]

    # Analyze last half for steady oscillation
    tail = outputs[sim_steps // 2 :]
    centered = tail - np.mean(tail)
    peak_amplitude = (np.max(tail) - np.min(tail)) / 2.0

    # Estimate period from zero crossings
    crossings = np.where(np.diff(np.sign(centered)))[0]
    if len(crossings) >= 2:
        avg_half_period = np.mean(np.diff(crossings)) * dt
        tu = 2.0 * avg_half_period
    else:
        tu = 1.0

    # Ultimate gain from describing function analysis
    ku = 4.0 * relay_amplitude / (math.pi * max(peak_amplitude, 1e-6))

    gains = ziegler_nichols_table(ku, tu, mode="pid")
    return TuningResult(gains=gains, ultimate_gain=ku, ultimate_period=tu, method="relay")


def _default_plant(state: NDArray, u: float, dt: float) -> NDArray:
    """Second-order underdamped plant for tuning demos."""
    wn, zeta = 4.0, 0.3
    x, xdot = state
    xddot = wn**2 * (u - x) - 2 * zeta * wn * xdot
    return np.array([x + xdot * dt, xdot + xddot * dt])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PID auto-tuning demo")
    parser.add_argument(
        "--method",
        choices=["ziegler-nichols", "relay"],
        default="relay",
        help="Tuning method",
    )
    args = parser.parse_args()

    if args.method == "ziegler-nichols":
        ku, tu = find_ultimate_gain(_default_plant)
        gains = ziegler_nichols_table(ku, tu, mode="pid")
        print(f"Ziegler-Nichols tuning:")
        print(f"  Ultimate gain Ku = {ku:.3f}")
        print(f"  Ultimate period Tu = {tu:.3f}s")
    else:
        result = relay_feedback_tuning(_default_plant)
        gains = result.gains
        print(f"Relay feedback tuning:")
        print(f"  Ultimate gain Ku = {result.ultimate_gain:.3f}")
        print(f"  Ultimate period Tu = {result.ultimate_period:.3f}s")

    print(f"  Kp = {gains.kp:.4f}")
    print(f"  Ki = {gains.ki:.4f}")
    print(f"  Kd = {gains.kd:.4f}")

    # Validate with step response
    config = PIDConfig(gains=gains, dt=0.01, output_min=-10.0, output_max=10.0)
    ctrl = PIDController(config)
    times, outputs, _ = ctrl.step_response(1.0, num_steps=500, plant_fn=_default_plant)
    ss_error = abs(1.0 - outputs[-1])
    print(f"  Validation steady-state error: {ss_error:.4f}")
