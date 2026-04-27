"""Geometric collision checking for circles and convex polygons.

Provides fast collision detection primitives used by the obstacle avoidance
modules. Supports point-circle, circle-circle, point-polygon, and
circle-polygon intersection tests, plus signed distance computation.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Circle:
    """A circular obstacle.

    Args:
        center: Center position [x, y].
        radius: Circle radius.
    """

    center: NDArray[np.float64]
    radius: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)


@dataclass
class ConvexPolygon:
    """A convex polygon obstacle defined by vertices in CCW order.

    Args:
        vertices: Array of shape (n, 2) with vertices in counter-clockwise order.
    """

    vertices: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)

    @property
    def edges(self) -> list[tuple[NDArray, NDArray]]:
        """Return list of (start, end) edge pairs."""
        n = len(self.vertices)
        return [(self.vertices[i], self.vertices[(i + 1) % n]) for i in range(n)]

    @property
    def centroid(self) -> NDArray[np.float64]:
        """Compute polygon centroid."""
        return np.mean(self.vertices, axis=0)


def point_in_circle(point: NDArray[np.float64], circle: Circle) -> bool:
    """Check if a point is inside a circle.

    Args:
        point: Query point [x, y].
        circle: Circle obstacle.

    Returns:
        True if point is inside or on the circle boundary.
    """
    return float(np.linalg.norm(point - circle.center)) <= circle.radius


def circle_circle_collision(c1: Circle, c2: Circle) -> bool:
    """Check if two circles overlap.

    Args:
        c1: First circle.
        c2: Second circle.

    Returns:
        True if circles overlap.
    """
    dist = float(np.linalg.norm(c1.center - c2.center))
    return dist <= c1.radius + c2.radius


def signed_distance_circle(point: NDArray[np.float64], circle: Circle) -> float:
    """Compute signed distance from a point to a circle boundary.

    Negative inside, positive outside.

    Args:
        point: Query point [x, y].
        circle: Circle obstacle.

    Returns:
        Signed distance (negative = inside).
    """
    return float(np.linalg.norm(point - circle.center)) - circle.radius


def point_to_segment_distance(
    point: NDArray[np.float64],
    seg_start: NDArray[np.float64],
    seg_end: NDArray[np.float64],
) -> float:
    """Compute minimum distance from a point to a line segment.

    Args:
        point: Query point.
        seg_start: Segment start point.
        seg_end: Segment end point.

    Returns:
        Minimum distance.
    """
    seg = seg_end - seg_start
    seg_len_sq = float(np.dot(seg, seg))

    if seg_len_sq < 1e-12:
        return float(np.linalg.norm(point - seg_start))

    t = float(np.dot(point - seg_start, seg)) / seg_len_sq
    t = max(0.0, min(1.0, t))
    projection = seg_start + t * seg
    return float(np.linalg.norm(point - projection))


def point_in_convex_polygon(point: NDArray[np.float64], polygon: ConvexPolygon) -> bool:
    """Check if a point is inside a convex polygon using cross products.

    Args:
        point: Query point [x, y].
        polygon: Convex polygon (vertices in CCW order).

    Returns:
        True if point is inside or on the polygon boundary.
    """
    n = len(polygon.vertices)
    for i in range(n):
        v1 = polygon.vertices[i]
        v2 = polygon.vertices[(i + 1) % n]
        edge = v2 - v1
        to_point = point - v1
        cross = edge[0] * to_point[1] - edge[1] * to_point[0]
        if cross < -1e-10:
            return False
    return True


def signed_distance_polygon(point: NDArray[np.float64], polygon: ConvexPolygon) -> float:
    """Compute signed distance from a point to a convex polygon.

    Negative inside, positive outside.

    Args:
        point: Query point [x, y].
        polygon: Convex polygon.

    Returns:
        Signed distance.
    """
    min_dist = float("inf")
    for seg_start, seg_end in polygon.edges:
        d = point_to_segment_distance(point, seg_start, seg_end)
        min_dist = min(min_dist, d)

    if point_in_convex_polygon(point, polygon):
        return -min_dist
    return min_dist


def circle_polygon_collision(circle: Circle, polygon: ConvexPolygon) -> bool:
    """Check if a circle intersects a convex polygon.

    Args:
        circle: Circle obstacle.
        polygon: Convex polygon obstacle.

    Returns:
        True if they overlap.
    """
    # Check if circle center is inside polygon
    if point_in_convex_polygon(circle.center, polygon):
        return True

    # Check if any polygon edge is within circle radius
    for seg_start, seg_end in polygon.edges:
        if point_to_segment_distance(circle.center, seg_start, seg_end) <= circle.radius:
            return True

    return False


def swept_circle_collision(
    p_start: NDArray[np.float64],
    p_end: NDArray[np.float64],
    robot_radius: float,
    obstacle: Circle,
    num_checks: int = 20,
) -> bool:
    """Check collision along a linear path with a circular robot.

    Args:
        p_start: Start position.
        p_end: End position.
        robot_radius: Robot collision radius.
        obstacle: Circular obstacle.
        num_checks: Number of interpolation points.

    Returns:
        True if any point along the path collides.
    """
    for i in range(num_checks + 1):
        t = i / num_checks
        p = p_start + t * (p_end - p_start)
        if signed_distance_circle(p, obstacle) <= robot_radius:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision checking demo")
    args = parser.parse_args()

    # Demo: circle obstacles
    obs1 = Circle(center=np.array([5.0, 5.0]), radius=1.0)
    obs2 = Circle(center=np.array([8.0, 3.0]), radius=0.5)

    test_points = [
        np.array([5.0, 5.0]),   # inside obs1
        np.array([6.5, 5.0]),   # outside obs1
        np.array([3.0, 3.0]),   # far from both
        np.array([8.2, 3.1]),   # inside obs2
    ]

    print("Circle collision tests:")
    for pt in test_points:
        d1 = signed_distance_circle(pt, obs1)
        d2 = signed_distance_circle(pt, obs2)
        print(f"  Point ({pt[0]:.1f}, {pt[1]:.1f}): "
              f"d(obs1)={d1:+.2f}, d(obs2)={d2:+.2f}")

    # Demo: polygon obstacle
    square = ConvexPolygon(vertices=np.array([
        [2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0],
    ]))

    print("\nPolygon collision tests:")
    for pt in [np.array([3.0, 3.0]), np.array([5.0, 3.0]), np.array([2.5, 2.5])]:
        inside = point_in_convex_polygon(pt, square)
        sd = signed_distance_polygon(pt, square)
        print(f"  Point ({pt[0]:.1f}, {pt[1]:.1f}): inside={inside}, sd={sd:+.3f}")

    # Demo: swept collision
    print("\nSwept collision test:")
    collides = swept_circle_collision(
        np.array([0.0, 5.0]), np.array([10.0, 5.0]), 0.3, obs1,
    )
    print(f"  Path (0,5)→(10,5) with r=0.3 vs obs1: collision={collides}")
