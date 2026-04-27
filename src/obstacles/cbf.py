"""Control Barrier Functions (CBFs) for provably safe obstacle avoidance.

CBFs provide formal safety guarantees by constraining the control input
so the system never enters unsafe states. Given a barrier function h(x)
where h(x) > 0 defines the safe set, the CBF condition is:

    dh/dt(x, u) + alpha(h(x)) >= 0

This ensures the system remains in the safe set for all time. The
implementation modifies a nominal controller by solving a QP that
finds the closest safe control input.

Reference: Ames et al. (2019), "Control Barrier Functions: Theory and Applications."
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .collision import Circle


@dataclass
class CBFConfig:
    """Configuration for the CBF safety filter.

    Args:
        alpha: CBF class-K function gain. Higher values allow the system
            to approach obstacles more closely before correcting.
        safety_margin: Additional safety margin beyond obstacle radius.
        u_max: Maximum control input magnitude.
    """

    alpha: float = 1.0
    safety_margin: float = 0.3
    u_max: float = 5.0


def barrier_function(
    pos: NDArray[np.float64], obstacle: Circle, safety_margin: float = 0.0
) -> float:
    """Compute the barrier function value h(x) for a circular obstacle.

    h(x) = ||p - p_obs||^2 - (r + margin)^2

    h(x) > 0 means safe, h(x) <= 0 means unsafe.

    Args:
        pos: Robot position [x, y].
        obstacle: Circular obstacle.
        safety_margin: Additional safety buffer.

    Returns:
        Barrier function value.
    """
    diff = pos - obstacle.center
    return float(np.dot(diff, diff)) - (obstacle.radius + safety_margin) ** 2


def barrier_gradient(
    pos: NDArray[np.float64], obstacle: Circle
) -> NDArray[np.float64]:
    """Compute the gradient of h(x) with respect to position.

    dh/dp = 2 * (p - p_obs)

    Args:
        pos: Robot position [x, y].
        obstacle: Circular obstacle.

    Returns:
        Gradient vector.
    """
    return 2.0 * (pos - obstacle.center)


class CBFSafetyFilter:
    """CBF-based safety filter that modifies a nominal control input.

    Given a nominal (desired) control input, the filter finds the closest
    safe input that satisfies the CBF constraint for all obstacles.

    For a single-integrator system (dx/dt = u), the CBF condition becomes:
        dh/dp @ u + alpha * h(x) >= 0

    For multiple obstacles, all constraints must be satisfied simultaneously.

    Args:
        obstacles: List of circular obstacles.
        config: CBF configuration.
    """

    def __init__(
        self, obstacles: list[Circle], config: CBFConfig | None = None
    ) -> None:
        self.obstacles = obstacles
        self.config = config or CBFConfig()

    def filter_control(
        self,
        pos: NDArray[np.float64],
        u_nominal: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply CBF safety filter to a nominal control input.

        Solves: min ||u - u_nom||^2
                s.t. dh_i/dp @ u + alpha * h_i(x) >= 0  for all i
                     ||u|| <= u_max

        Uses iterative projection for the multi-constraint case.

        Args:
            pos: Current robot position.
            u_nominal: Desired control input.

        Returns:
            Safe control input (closest to nominal that satisfies all CBF constraints).
        """
        cfg = self.config
        u = u_nominal.copy()

        # Iterative projection onto CBF constraint half-spaces
        for _ in range(10):
            all_satisfied = True
            for obs in self.obstacles:
                h = barrier_function(pos, obs, cfg.safety_margin)
                grad_h = barrier_gradient(pos, obs)

                # CBF condition: grad_h @ u + alpha * h >= 0
                constraint_val = float(np.dot(grad_h, u)) + cfg.alpha * h

                if constraint_val < 0:
                    all_satisfied = False
                    # Project u onto the constraint boundary
                    grad_norm_sq = float(np.dot(grad_h, grad_h))
                    if grad_norm_sq > 1e-10:
                        correction = (-constraint_val / grad_norm_sq) * grad_h
                        u = u + correction

            if all_satisfied:
                break

        # Enforce input magnitude constraint
        u_norm = np.linalg.norm(u)
        if u_norm > cfg.u_max:
            u = u * (cfg.u_max / u_norm)

        return u

    def is_safe(self, pos: NDArray[np.float64]) -> bool:
        """Check if the current position is in the safe set.

        Args:
            pos: Robot position.

        Returns:
            True if h(x) > 0 for all obstacles.
        """
        for obs in self.obstacles:
            if barrier_function(pos, obs, self.config.safety_margin) <= 0:
                return False
        return True


def simulate_with_cbf(
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    obstacles: list[Circle],
    config: CBFConfig | None = None,
    dt: float = 0.05,
    num_steps: int = 300,
    k_goal: float = 1.0,
) -> tuple[NDArray[np.float64], list[bool]]:
    """Simulate a point robot navigating to a goal with CBF safety.

    Uses a simple proportional controller as the nominal policy,
    filtered through the CBF for safety.

    Args:
        start: Start position [x, y].
        goal: Goal position [x, y].
        obstacles: List of circular obstacles.
        config: CBF configuration.
        dt: Simulation timestep.
        num_steps: Number of simulation steps.
        k_goal: Proportional gain for goal-seeking.

    Returns:
        Tuple of (trajectory array (n, 2), safety_flags list).
    """
    cbf = CBFSafetyFilter(obstacles, config)
    pos = np.array(start, dtype=np.float64)
    trajectory = [pos.copy()]
    safety_flags: list[bool] = []

    for _ in range(num_steps):
        # Nominal controller: go toward goal
        u_nominal = k_goal * (goal - pos)

        # Apply CBF safety filter
        u_safe = cbf.filter_control(pos, u_nominal)

        # Single integrator dynamics
        pos = pos + u_safe * dt
        trajectory.append(pos.copy())
        safety_flags.append(cbf.is_safe(pos))

        # Check if goal reached
        if np.linalg.norm(pos - goal) < 0.1:
            break

    return np.array(trajectory), safety_flags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBF obstacle avoidance demo")
    parser.add_argument(
        "--obstacles", type=str, default="4,6,1.0;7,3,0.8",
        help="Obstacles as 'x,y,r;...'",
    )
    parser.add_argument("--goal", type=str, default="10,10", help="Goal 'x,y'")
    parser.add_argument("--alpha", type=float, default=1.0, help="CBF alpha gain")
    args = parser.parse_args()

    goal = np.array([float(c) for c in args.goal.split(",")])
    obstacles = []
    for obs_str in args.obstacles.split(";"):
        parts = [float(p) for p in obs_str.strip().split(",")]
        obstacles.append(Circle(center=np.array(parts[:2]), radius=parts[2]))

    config = CBFConfig(alpha=args.alpha, safety_margin=0.3, u_max=3.0)
    traj, safety = simulate_with_cbf(
        start=np.array([0.0, 0.0]),
        goal=goal,
        obstacles=obstacles,
        config=config,
    )

    print(f"CBF obstacle avoidance")
    print(f"  Goal: ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Path length: {len(traj)} steps")
    print(f"  Always safe: {all(safety)}")
    print()

    step = max(1, len(traj) // 10)
    for i in range(0, len(traj), step):
        d_goal = np.linalg.norm(traj[i] - goal)
        min_obs_dist = min(
            np.linalg.norm(traj[i] - obs.center) - obs.radius
            for obs in obstacles
        )
        print(f"  Step {i:3d}: ({traj[i, 0]:.2f}, {traj[i, 1]:.2f})  "
              f"d_goal={d_goal:.2f}  min_obs_dist={min_obs_dist:.2f}")

    final_dist = np.linalg.norm(traj[-1] - goal)
    print(f"\n  Final distance to goal: {final_dist:.3f}")
