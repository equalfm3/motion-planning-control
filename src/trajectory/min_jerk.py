"""Minimum-jerk trajectory generation.

Generates smooth trajectories that minimize the integral of squared jerk
(third derivative of position). The solution is a 5th-order polynomial
with coefficients determined by boundary conditions on position, velocity,
and acceleration at the start and end of the motion.

Reference: Flash & Hogan (1985), "The Coordination of Arm Movements."
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class BoundaryCondition:
    """Boundary conditions for a single dimension.

    Args:
        pos: Position.
        vel: Velocity.
        acc: Acceleration.
    """

    pos: float = 0.0
    vel: float = 0.0
    acc: float = 0.0


@dataclass
class TrajectoryPoint:
    """A single point along a trajectory.

    Args:
        time: Timestamp.
        position: Position vector.
        velocity: Velocity vector.
        acceleration: Acceleration vector.
        jerk: Jerk vector.
    """

    time: float
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    jerk: NDArray[np.float64]


def min_jerk_coefficients(
    bc_start: BoundaryCondition,
    bc_end: BoundaryCondition,
    duration: float,
) -> NDArray[np.float64]:
    """Compute 5th-order polynomial coefficients for minimum-jerk trajectory.

    The polynomial is: x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5

    Args:
        bc_start: Boundary conditions at t=0.
        bc_end: Boundary conditions at t=duration.
        duration: Total trajectory duration.

    Returns:
        Coefficient array [a0, a1, a2, a3, a4, a5].
    """
    T = duration
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5

    a0 = bc_start.pos
    a1 = bc_start.vel
    a2 = bc_start.acc / 2.0

    # Solve for a3, a4, a5 from end boundary conditions
    # x(T) = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4 + a5*T^5
    # x'(T) = a1 + 2*a2*T + 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4
    # x''(T) = 2*a2 + 6*a3*T + 12*a4*T^2 + 20*a5*T^3
    M = np.array([
        [T3, T4, T5],
        [3 * T2, 4 * T3, 5 * T4],
        [6 * T, 12 * T2, 20 * T3],
    ])
    rhs = np.array([
        bc_end.pos - a0 - a1 * T - a2 * T2,
        bc_end.vel - a1 - 2 * a2 * T,
        bc_end.acc - 2 * a2,
    ])
    a345 = np.linalg.solve(M, rhs)

    return np.array([a0, a1, a2, a345[0], a345[1], a345[2]])


def evaluate_polynomial(
    coeffs: NDArray[np.float64], t: float
) -> tuple[float, float, float, float]:
    """Evaluate a 5th-order polynomial and its derivatives.

    Args:
        coeffs: Polynomial coefficients [a0, ..., a5].
        t: Time value.

    Returns:
        Tuple of (position, velocity, acceleration, jerk).
    """
    a = coeffs
    pos = a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3 + a[4] * t**4 + a[5] * t**5
    vel = a[1] + 2 * a[2] * t + 3 * a[3] * t**2 + 4 * a[4] * t**3 + 5 * a[5] * t**4
    acc = 2 * a[2] + 6 * a[3] * t + 12 * a[4] * t**2 + 20 * a[5] * t**3
    jerk = 6 * a[3] + 24 * a[4] * t + 60 * a[5] * t**2
    return pos, vel, acc, jerk


def generate_min_jerk_trajectory(
    waypoints: NDArray[np.float64],
    duration: float,
    num_samples: int = 100,
) -> list[TrajectoryPoint]:
    """Generate a minimum-jerk trajectory through waypoints.

    For multiple waypoints, the trajectory is split into equal-duration
    segments, each a minimum-jerk polynomial with zero velocity and
    acceleration at the endpoints (rest-to-rest).

    Args:
        waypoints: Array of shape (n_waypoints, n_dims).
        duration: Total trajectory duration in seconds.
        num_samples: Number of output samples.

    Returns:
        List of trajectory points.
    """
    waypoints = np.atleast_2d(waypoints)
    n_waypoints, n_dims = waypoints.shape
    n_segments = n_waypoints - 1

    if n_segments < 1:
        raise ValueError("Need at least 2 waypoints")

    seg_duration = duration / n_segments
    dt = duration / (num_samples - 1)

    # Compute coefficients for each segment and dimension
    all_coeffs: list[list[NDArray]] = []
    for seg in range(n_segments):
        seg_coeffs = []
        for dim in range(n_dims):
            bc_start = BoundaryCondition(pos=waypoints[seg, dim])
            bc_end = BoundaryCondition(pos=waypoints[seg + 1, dim])
            coeffs = min_jerk_coefficients(bc_start, bc_end, seg_duration)
            seg_coeffs.append(coeffs)
        all_coeffs.append(seg_coeffs)

    # Sample the trajectory
    trajectory: list[TrajectoryPoint] = []
    for i in range(num_samples):
        t_global = i * dt
        seg_idx = min(int(t_global / seg_duration), n_segments - 1)
        t_local = t_global - seg_idx * seg_duration

        pos = np.zeros(n_dims)
        vel = np.zeros(n_dims)
        acc = np.zeros(n_dims)
        jerk = np.zeros(n_dims)

        for dim in range(n_dims):
            p, v, a, j = evaluate_polynomial(all_coeffs[seg_idx][dim], t_local)
            pos[dim] = p
            vel[dim] = v
            acc[dim] = a
            jerk[dim] = j

        trajectory.append(TrajectoryPoint(
            time=t_global, position=pos, velocity=vel,
            acceleration=acc, jerk=jerk,
        ))

    return trajectory


def parse_waypoints(waypoint_str: str) -> NDArray[np.float64]:
    """Parse waypoint string like '0,0;5,3;10,0' into array.

    Args:
        waypoint_str: Semicolon-separated waypoints, comma-separated coords.

    Returns:
        Waypoint array of shape (n_waypoints, n_dims).
    """
    points = []
    for wp in waypoint_str.split(";"):
        coords = [float(c) for c in wp.strip().split(",")]
        points.append(coords)
    return np.array(points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimum-jerk trajectory demo")
    parser.add_argument(
        "--waypoints", type=str, default="0,0;5,3;10,0",
        help="Waypoints as 'x1,y1;x2,y2;...'",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Total duration (s)")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    args = parser.parse_args()

    waypoints = parse_waypoints(args.waypoints)
    traj = generate_min_jerk_trajectory(waypoints, args.duration, args.samples)

    print(f"Minimum-jerk trajectory ({len(waypoints)} waypoints, {args.duration}s)")
    print(f"  Waypoints: {waypoints.tolist()}")
    print()

    for i, pt in enumerate(traj):
        if i % (len(traj) // 10) == 0 or i == len(traj) - 1:
            pos_str = ", ".join(f"{v:.3f}" for v in pt.position)
            vel_str = ", ".join(f"{v:.3f}" for v in pt.velocity)
            print(f"  t={pt.time:.2f}s  pos=({pos_str})  vel=({vel_str})")

    # Verify smoothness: compute total jerk
    total_jerk = sum(np.linalg.norm(pt.jerk) for pt in traj)
    print(f"\n  Total jerk magnitude: {total_jerk:.3f}")
