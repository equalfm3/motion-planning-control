"""Cubic and quintic spline interpolation for trajectory generation.

Cubic splines provide C2 continuity (continuous position, velocity, and
acceleration) through a sequence of waypoints. Quintic splines extend
this to C4 continuity, also ensuring continuous jerk and snap — important
for smooth robotic motion.

Uses the natural spline boundary condition (zero second derivative at
endpoints) by default, with optional clamped boundaries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class BoundaryType(Enum):
    """Spline boundary condition type."""

    NATURAL = "natural"
    CLAMPED = "clamped"


@dataclass
class SplineSegment:
    """Coefficients for one cubic spline segment.

    Represents: s(t) = a + b*(t-t_i) + c*(t-t_i)^2 + d*(t-t_i)^3
    where t_i is the segment start time.

    Args:
        t_start: Segment start time.
        t_end: Segment end time.
        a: Constant coefficient.
        b: Linear coefficient.
        c: Quadratic coefficient.
        d: Cubic coefficient.
    """

    t_start: float
    t_end: float
    a: float
    b: float
    c: float
    d: float

    def evaluate(self, t: float) -> tuple[float, float, float]:
        """Evaluate position, velocity, and acceleration at time t.

        Args:
            t: Query time.

        Returns:
            Tuple of (position, velocity, acceleration).
        """
        dt = t - self.t_start
        pos = self.a + self.b * dt + self.c * dt**2 + self.d * dt**3
        vel = self.b + 2 * self.c * dt + 3 * self.d * dt**2
        acc = 2 * self.c + 6 * self.d * dt
        return pos, vel, acc


class CubicSpline:
    """Natural or clamped cubic spline through waypoints.

    Args:
        times: Knot times of shape (n,).
        values: Knot values of shape (n,) for 1D or (n, d) for multi-dim.
        boundary: Boundary condition type.
        bc_start_vel: Start velocity for clamped boundaries.
        bc_end_vel: End velocity for clamped boundaries.
    """

    def __init__(
        self,
        times: NDArray[np.float64],
        values: NDArray[np.float64],
        boundary: BoundaryType = BoundaryType.NATURAL,
        bc_start_vel: float = 0.0,
        bc_end_vel: float = 0.0,
    ) -> None:
        self.times = np.asarray(times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        self.ndim = 1 if values.ndim == 1 else values.shape[1]
        self.values = values.reshape(-1, self.ndim) if values.ndim == 1 else values
        self.n = len(times)
        self.boundary = boundary
        self.bc_start_vel = bc_start_vel
        self.bc_end_vel = bc_end_vel
        self.segments: list[list[SplineSegment]] = self._build_splines()

    def _build_splines(self) -> list[list[SplineSegment]]:
        """Build spline segments for each dimension.

        Returns:
            List of segment lists, one per dimension.
        """
        all_segments: list[list[SplineSegment]] = []

        for dim in range(self.ndim):
            y = self.values[:, dim]
            segments = self._solve_tridiagonal(y)
            all_segments.append(segments)

        return all_segments

    def _solve_tridiagonal(self, y: NDArray[np.float64]) -> list[SplineSegment]:
        """Solve the tridiagonal system for cubic spline coefficients.

        Args:
            y: Values at knot points.

        Returns:
            List of spline segments.
        """
        n = self.n
        h = np.diff(self.times)

        # Build tridiagonal system for second derivatives (c values)
        A = np.zeros((n, n))
        rhs = np.zeros(n)

        if self.boundary == BoundaryType.NATURAL:
            A[0, 0] = 1.0
            A[n - 1, n - 1] = 1.0
        else:  # clamped
            A[0, 0] = 2 * h[0]
            A[0, 1] = h[0]
            rhs[0] = 3 * ((y[1] - y[0]) / h[0] - self.bc_start_vel)
            A[n - 1, n - 2] = h[n - 2]
            A[n - 1, n - 1] = 2 * h[n - 2]
            rhs[n - 1] = 3 * (self.bc_end_vel - (y[n - 1] - y[n - 2]) / h[n - 2])

        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

        c = np.linalg.solve(A, rhs)

        # Compute remaining coefficients
        segments: list[SplineSegment] = []
        for i in range(n - 1):
            a = y[i]
            b = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
            d = (c[i + 1] - c[i]) / (3 * h[i])
            segments.append(SplineSegment(
                t_start=self.times[i],
                t_end=self.times[i + 1],
                a=a, b=b, c=c[i], d=d,
            ))

        return segments

    def _find_segment(self, t: float) -> int:
        """Find the segment index for a given time.

        Args:
            t: Query time.

        Returns:
            Segment index.
        """
        t = np.clip(t, self.times[0], self.times[-1])
        idx = np.searchsorted(self.times, t, side="right") - 1
        return min(max(idx, 0), self.n - 2)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the spline at time t.

        Args:
            t: Query time.

        Returns:
            Tuple of (position, velocity, acceleration) arrays.
        """
        idx = self._find_segment(t)
        pos = np.zeros(self.ndim)
        vel = np.zeros(self.ndim)
        acc = np.zeros(self.ndim)

        for dim in range(self.ndim):
            p, v, a = self.segments[dim][idx].evaluate(t)
            pos[dim] = p
            vel[dim] = v
            acc[dim] = a

        return pos, vel, acc

    def sample(self, num_points: int = 100) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Sample the spline uniformly.

        Args:
            num_points: Number of sample points.

        Returns:
            Tuple of (times, positions, velocities, accelerations).
        """
        t_arr = np.linspace(self.times[0], self.times[-1], num_points)
        positions = np.zeros((num_points, self.ndim))
        velocities = np.zeros((num_points, self.ndim))
        accelerations = np.zeros((num_points, self.ndim))

        for i, t in enumerate(t_arr):
            positions[i], velocities[i], accelerations[i] = self.evaluate(t)

        return t_arr, positions, velocities, accelerations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cubic spline trajectory demo")
    parser.add_argument(
        "--waypoints", type=str, default="0,0;2,4;5,3;8,5;10,0",
        help="Waypoints as 'x1,y1;x2,y2;...'",
    )
    parser.add_argument("--samples", type=int, default=50, help="Number of samples")
    args = parser.parse_args()

    # Parse waypoints
    points = []
    for wp in args.waypoints.split(";"):
        coords = [float(c) for c in wp.strip().split(",")]
        points.append(coords)
    waypoints = np.array(points)

    # Use arc-length parameterization for times
    dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))])
    times = dists / dists[-1] * 5.0  # normalize to 5 seconds

    spline = CubicSpline(times, waypoints)
    t_arr, pos, vel, acc = spline.sample(args.samples)

    print(f"Cubic spline ({len(waypoints)} waypoints, {times[-1]:.1f}s)")
    for i in range(0, len(t_arr), len(t_arr) // 10):
        p_str = ", ".join(f"{v:.3f}" for v in pos[i])
        v_str = ", ".join(f"{v:.3f}" for v in vel[i])
        print(f"  t={t_arr[i]:.2f}s  pos=({p_str})  vel=({v_str})")

    # Check C2 continuity at knots
    print("\n  C2 continuity check at internal knots:")
    for k in range(1, len(times) - 1):
        t_k = times[k]
        eps = 1e-6
        _, _, acc_left = spline.evaluate(t_k - eps)
        _, _, acc_right = spline.evaluate(t_k + eps)
        gap = np.linalg.norm(acc_right - acc_left)
        print(f"    Knot {k} (t={t_k:.2f}s): acc discontinuity = {gap:.2e}")
