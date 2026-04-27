"""Time-optimal trajectory generation with velocity and acceleration limits.

Computes the fastest trajectory along a given path subject to maximum
velocity and acceleration constraints. Uses a bang-bang acceleration
profile (maximum acceleration → cruise → maximum deceleration) which
is provably time-optimal for point-to-point motion with symmetric
velocity and acceleration bounds.

For multi-segment paths, applies a trapezoidal velocity profile to each
segment with proper velocity continuity at waypoints.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class VelocityProfile:
    """Trapezoidal velocity profile for one segment.

    Args:
        t_accel: Duration of acceleration phase.
        t_cruise: Duration of cruise (constant velocity) phase.
        t_decel: Duration of deceleration phase.
        v_peak: Peak velocity reached.
        distance: Total distance of the segment.
    """

    t_accel: float
    t_cruise: float
    t_decel: float
    v_peak: float
    distance: float

    @property
    def total_time(self) -> float:
        """Total segment duration."""
        return self.t_accel + self.t_cruise + self.t_decel


def compute_trapezoidal_profile(
    distance: float,
    v_max: float,
    a_max: float,
    v_start: float = 0.0,
    v_end: float = 0.0,
) -> VelocityProfile:
    """Compute a trapezoidal velocity profile for a single segment.

    If the distance is too short to reach v_max, the profile becomes
    triangular (no cruise phase).

    Args:
        distance: Segment length (positive).
        v_max: Maximum allowed velocity.
        a_max: Maximum allowed acceleration/deceleration.
        v_start: Initial velocity.
        v_end: Final velocity.

    Returns:
        Velocity profile for the segment.
    """
    distance = abs(distance)
    if distance < 1e-10:
        return VelocityProfile(0.0, 0.0, 0.0, 0.0, 0.0)

    # Time and distance to accelerate from v_start to v_max
    t_accel_full = (v_max - v_start) / a_max
    d_accel_full = (v_max**2 - v_start**2) / (2 * a_max)

    # Time and distance to decelerate from v_max to v_end
    t_decel_full = (v_max - v_end) / a_max
    d_decel_full = (v_max**2 - v_end**2) / (2 * a_max)

    if d_accel_full + d_decel_full <= distance:
        # Trapezoidal profile: accel → cruise → decel
        d_cruise = distance - d_accel_full - d_decel_full
        t_cruise = d_cruise / v_max
        return VelocityProfile(t_accel_full, t_cruise, t_decel_full, v_max, distance)
    else:
        # Triangular profile: can't reach v_max
        # v_peak^2 = v_start^2 + 2*a_max*d_accel
        # v_peak^2 = v_end^2 + 2*a_max*d_decel
        # d_accel + d_decel = distance
        v_peak = np.sqrt((2 * a_max * distance + v_start**2 + v_end**2) / 2)
        v_peak = min(v_peak, v_max)
        t_accel = (v_peak - v_start) / a_max
        t_decel = (v_peak - v_end) / a_max
        return VelocityProfile(t_accel, 0.0, t_decel, v_peak, distance)


def evaluate_profile(
    profile: VelocityProfile,
    t: float,
    v_start: float = 0.0,
    a_max: float = 1.0,
) -> tuple[float, float, float]:
    """Evaluate position, velocity, and acceleration at time t within a profile.

    Args:
        profile: Velocity profile.
        t: Time within the segment.
        v_start: Initial velocity.
        a_max: Maximum acceleration.

    Returns:
        Tuple of (position, velocity, acceleration).
    """
    t = max(0.0, min(t, profile.total_time))

    if t <= profile.t_accel:
        # Acceleration phase
        acc = a_max
        vel = v_start + acc * t
        pos = v_start * t + 0.5 * acc * t**2
    elif t <= profile.t_accel + profile.t_cruise:
        # Cruise phase
        dt = t - profile.t_accel
        d_accel = v_start * profile.t_accel + 0.5 * a_max * profile.t_accel**2
        acc = 0.0
        vel = profile.v_peak
        pos = d_accel + profile.v_peak * dt
    else:
        # Deceleration phase
        dt = t - profile.t_accel - profile.t_cruise
        d_accel = v_start * profile.t_accel + 0.5 * a_max * profile.t_accel**2
        d_cruise = profile.v_peak * profile.t_cruise
        acc = -a_max
        vel = profile.v_peak + acc * dt
        pos = d_accel + d_cruise + profile.v_peak * dt + 0.5 * acc * dt**2

    return pos, vel, acc


def generate_time_optimal_trajectory(
    waypoints: NDArray[np.float64],
    v_max: float = 2.0,
    a_max: float = 1.0,
    num_samples: int = 100,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate a time-optimal trajectory through waypoints.

    Computes trapezoidal velocity profiles for each segment and
    interpolates along the path.

    Args:
        waypoints: Array of shape (n_waypoints, n_dims).
        v_max: Maximum velocity.
        a_max: Maximum acceleration.
        num_samples: Number of output samples.

    Returns:
        Tuple of (times, positions, velocities, accelerations).
    """
    waypoints = np.atleast_2d(waypoints)
    n_wp, n_dims = waypoints.shape
    n_seg = n_wp - 1

    # Compute segment distances and directions
    segments: list[tuple[float, NDArray]] = []
    for i in range(n_seg):
        diff = waypoints[i + 1] - waypoints[i]
        dist = np.linalg.norm(diff)
        direction = diff / dist if dist > 1e-10 else np.zeros(n_dims)
        segments.append((dist, direction))

    # Compute velocity profiles (rest-to-rest at each waypoint)
    profiles: list[VelocityProfile] = []
    for dist, _ in segments:
        profile = compute_trapezoidal_profile(dist, v_max, a_max)
        profiles.append(profile)

    # Total time
    total_time = sum(p.total_time for p in profiles)
    if total_time < 1e-10:
        t_arr = np.zeros(num_samples)
        return t_arr, np.tile(waypoints[0], (num_samples, 1)), np.zeros((num_samples, n_dims)), np.zeros((num_samples, n_dims))

    dt = total_time / (num_samples - 1)
    t_arr = np.arange(num_samples) * dt
    positions = np.zeros((num_samples, n_dims))
    velocities = np.zeros((num_samples, n_dims))
    accelerations = np.zeros((num_samples, n_dims))

    # Segment start times
    seg_starts = np.zeros(n_seg)
    for i in range(1, n_seg):
        seg_starts[i] = seg_starts[i - 1] + profiles[i - 1].total_time

    for i, t in enumerate(t_arr):
        # Find active segment
        seg_idx = n_seg - 1
        for s in range(n_seg):
            if t < seg_starts[s] + profiles[s].total_time:
                seg_idx = s
                break

        t_local = t - seg_starts[seg_idx]
        dist, direction = segments[seg_idx]
        profile = profiles[seg_idx]

        s_pos, s_vel, s_acc = evaluate_profile(profile, t_local, 0.0, a_max)

        positions[i] = waypoints[seg_idx] + direction * s_pos
        velocities[i] = direction * s_vel
        accelerations[i] = direction * s_acc

    return t_arr, positions, velocities, accelerations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time-optimal trajectory demo")
    parser.add_argument("--v-max", type=float, default=2.0, help="Max velocity")
    parser.add_argument("--a-max", type=float, default=1.0, help="Max acceleration")
    parser.add_argument(
        "--waypoints", type=str, default="0,0;5,3;10,0;10,5",
        help="Waypoints as 'x1,y1;x2,y2;...'",
    )
    args = parser.parse_args()

    points = []
    for wp in args.waypoints.split(";"):
        coords = [float(c) for c in wp.strip().split(",")]
        points.append(coords)
    waypoints = np.array(points)

    t_arr, pos, vel, acc = generate_time_optimal_trajectory(
        waypoints, args.v_max, args.a_max, num_samples=60,
    )

    total_time = t_arr[-1]
    print(f"Time-optimal trajectory (v_max={args.v_max}, a_max={args.a_max})")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Waypoints: {waypoints.tolist()}")
    print()

    for i in range(0, len(t_arr), len(t_arr) // 10):
        p_str = ", ".join(f"{v:.3f}" for v in pos[i])
        speed = np.linalg.norm(vel[i])
        print(f"  t={t_arr[i]:.2f}s  pos=({p_str})  speed={speed:.3f}")

    max_speed = max(np.linalg.norm(vel[i]) for i in range(len(t_arr)))
    max_acc = max(np.linalg.norm(acc[i]) for i in range(len(t_arr)))
    print(f"\n  Max speed: {max_speed:.3f} (limit: {args.v_max})")
    print(f"  Max accel: {max_acc:.3f} (limit: {args.a_max})")
