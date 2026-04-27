"""Artificial potential field obstacle avoidance.

Implements the classic potential field approach where the goal exerts an
attractive force and obstacles exert repulsive forces. The robot follows
the negative gradient of the total potential.

Known limitation: potential fields can have local minima where attractive
and repulsive forces cancel. This implementation includes a random
perturbation escape mechanism for local minima detection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .collision import Circle, signed_distance_circle


@dataclass
class PotentialFieldConfig:
    """Configuration for the potential field planner.

    Args:
        k_att: Attractive potential gain.
        k_rep: Repulsive potential gain.
        d_influence: Obstacle influence distance — repulsion is zero beyond this.
        step_size: Gradient descent step size.
        goal_tolerance: Distance threshold to consider goal reached.
        max_steps: Maximum number of gradient descent steps.
        local_min_threshold: Gradient norm below which local minimum is detected.
    """

    k_att: float = 1.0
    k_rep: float = 100.0
    d_influence: float = 3.0
    step_size: float = 0.1
    goal_tolerance: float = 0.1
    max_steps: int = 500
    local_min_threshold: float = 0.01


def attractive_potential(
    pos: NDArray[np.float64], goal: NDArray[np.float64], k_att: float
) -> float:
    """Compute quadratic attractive potential toward the goal.

    Args:
        pos: Current position.
        goal: Goal position.
        k_att: Attractive gain.

    Returns:
        Attractive potential value.
    """
    return 0.5 * k_att * float(np.sum((pos - goal) ** 2))


def attractive_force(
    pos: NDArray[np.float64], goal: NDArray[np.float64], k_att: float
) -> NDArray[np.float64]:
    """Compute attractive force (negative gradient of attractive potential).

    Args:
        pos: Current position.
        goal: Goal position.
        k_att: Attractive gain.

    Returns:
        Attractive force vector.
    """
    return -k_att * (pos - goal)


def repulsive_potential(
    pos: NDArray[np.float64],
    obstacle: Circle,
    k_rep: float,
    d_influence: float,
) -> float:
    """Compute repulsive potential from a single obstacle.

    Uses the inverse-distance formulation: U_rep = 0.5 * k_rep * (1/d - 1/d0)^2
    when d < d0, and 0 otherwise.

    Args:
        pos: Current position.
        obstacle: Circular obstacle.
        k_rep: Repulsive gain.
        d_influence: Influence distance.

    Returns:
        Repulsive potential value.
    """
    d = max(signed_distance_circle(pos, obstacle), 1e-6)
    if d >= d_influence:
        return 0.0
    return 0.5 * k_rep * (1.0 / d - 1.0 / d_influence) ** 2


def repulsive_force(
    pos: NDArray[np.float64],
    obstacle: Circle,
    k_rep: float,
    d_influence: float,
) -> NDArray[np.float64]:
    """Compute repulsive force from a single obstacle.

    Args:
        pos: Current position.
        obstacle: Circular obstacle.
        k_rep: Repulsive gain.
        d_influence: Influence distance.

    Returns:
        Repulsive force vector.
    """
    d = max(signed_distance_circle(pos, obstacle), 1e-6)
    if d >= d_influence:
        return np.zeros(2)

    # Gradient direction: away from obstacle
    direction = pos - obstacle.center
    dist_to_center = np.linalg.norm(direction)
    if dist_to_center < 1e-6:
        direction = np.random.randn(2)
        dist_to_center = np.linalg.norm(direction)
    direction = direction / dist_to_center

    magnitude = k_rep * (1.0 / d - 1.0 / d_influence) / (d**2)
    return magnitude * direction


def total_force(
    pos: NDArray[np.float64],
    goal: NDArray[np.float64],
    obstacles: list[Circle],
    config: PotentialFieldConfig,
) -> NDArray[np.float64]:
    """Compute total force (attractive + all repulsive).

    Args:
        pos: Current position.
        goal: Goal position.
        obstacles: List of circular obstacles.
        config: Potential field configuration.

    Returns:
        Total force vector.
    """
    f = attractive_force(pos, goal, config.k_att)
    for obs in obstacles:
        f += repulsive_force(pos, obs, config.k_rep, config.d_influence)
    return f


def plan_path(
    start: NDArray[np.float64],
    goal: NDArray[np.float64],
    obstacles: list[Circle],
    config: PotentialFieldConfig | None = None,
) -> tuple[NDArray[np.float64], bool]:
    """Plan a path using gradient descent on the potential field.

    Args:
        start: Start position [x, y].
        goal: Goal position [x, y].
        obstacles: List of circular obstacles.
        config: Planner configuration.

    Returns:
        Tuple of (path array of shape (n, 2), reached_goal boolean).
    """
    config = config or PotentialFieldConfig()
    pos = np.array(start, dtype=np.float64)
    path = [pos.copy()]
    rng = np.random.default_rng(42)

    for step in range(config.max_steps):
        if np.linalg.norm(pos - goal) < config.goal_tolerance:
            return np.array(path), True

        f = total_force(pos, goal, obstacles, config)
        f_norm = np.linalg.norm(f)

        # Local minimum escape: add random perturbation
        if f_norm < config.local_min_threshold:
            perturbation = rng.standard_normal(2) * config.step_size * 2
            pos = pos + perturbation
        else:
            # Normalize force and take a step
            direction = f / f_norm
            pos = pos + config.step_size * direction

        path.append(pos.copy())

    reached = np.linalg.norm(pos - goal) < config.goal_tolerance
    return np.array(path), reached


def parse_obstacles(obs_str: str) -> list[Circle]:
    """Parse obstacle string like '5,5,1.0;8,3,0.5' into Circle list.

    Args:
        obs_str: Semicolon-separated obstacles, each 'x,y,radius'.

    Returns:
        List of Circle obstacles.
    """
    obstacles = []
    for obs in obs_str.split(";"):
        parts = [float(p) for p in obs.strip().split(",")]
        obstacles.append(Circle(center=np.array(parts[:2]), radius=parts[2]))
    return obstacles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Potential field planner demo")
    parser.add_argument("--start", type=str, default="0,0", help="Start position 'x,y'")
    parser.add_argument("--goal", type=str, default="10,10", help="Goal position 'x,y'")
    parser.add_argument(
        "--obstacles", type=str, default="4,6,1.0;7,3,0.8",
        help="Obstacles as 'x,y,r;...'",
    )
    args = parser.parse_args()

    start = np.array([float(c) for c in args.start.split(",")])
    goal = np.array([float(c) for c in args.goal.split(",")])
    obstacles = parse_obstacles(args.obstacles)

    config = PotentialFieldConfig(k_att=2.0, k_rep=10.0, d_influence=2.0, local_min_threshold=0.05)
    path, reached = plan_path(start, goal, obstacles, config)

    print(f"Potential field path planning")
    print(f"  Start: ({start[0]:.1f}, {start[1]:.1f})")
    print(f"  Goal:  ({goal[0]:.1f}, {goal[1]:.1f})")
    print(f"  Obstacles: {len(obstacles)}")
    print(f"  Path length: {len(path)} steps")
    print(f"  Reached goal: {reached}")
    print()

    step = max(1, len(path) // 10)
    for i in range(0, len(path), step):
        d_goal = np.linalg.norm(path[i] - goal)
        print(f"  Step {i:3d}: ({path[i, 0]:.2f}, {path[i, 1]:.2f})  d_goal={d_goal:.2f}")
