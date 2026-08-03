import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class RRTNode:
    x: float
    y: float
    parent: Optional[int] = None
    cost: float = 0.0


@dataclass
class PlanningCostmap:
    width: int
    height: int
    resolution: float
    origin: Tuple[float, float, float]
    data: Sequence[int]
    occupied_threshold: int = 50

    @property
    def x_min(self) -> float:
        return self.origin[0]

    @property
    def y_min(self) -> float:
        return self.origin[1]

    @property
    def x_max(self) -> float:
        return self.origin[0] + self.width * self.resolution

    @property
    def y_max(self) -> float:
        return self.origin[1] + self.height * self.resolution

    def index(self, col: int, row: int) -> int:
        return row * self.width + col


@dataclass
class RRTStarResult:
    status: str
    path: List[Tuple[float, float]] = field(default_factory=list)
    nodes: List[RRTNode] = field(default_factory=list)
    edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=list)
    iterations: int = 0
    elapsed_time: float = 0.0


class RRTStarPlanner:
    def __init__(
        self,
        max_iterations: int = 3000,
        step_size: float = 0.30,
        goal_tolerance: float = 0.35,
        goal_sample_rate: float = 0.10,
        search_radius: float = 0.80,
        collision_check_resolution: float = 0.05,
        occupied_threshold: int = 50,
        planning_timeout: float = 5.0,
    ):
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.collision_check_resolution = collision_check_resolution
        self.occupied_threshold = occupied_threshold
        self.planning_timeout = planning_timeout

        self.costmap: Optional[PlanningCostmap] = None
        self.nodes: List[RRTNode] = []
        self.goal_xy: Tuple[float, float] = (0.0, 0.0)

    def plan(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        costmap: PlanningCostmap,
    ) -> RRTStarResult:
        self.costmap = costmap
        self.costmap.occupied_threshold = self.occupied_threshold
        self.nodes = []
        self.goal_xy = goal_xy
        start_time = time.monotonic()

        if not self._is_inside_world(start_xy[0], start_xy[1]):
            return self._result('NO_PLAN_START_OUTSIDE_MAP', 0, start_time)
        if not self._is_inside_world(goal_xy[0], goal_xy[1]):
            return self._result('NO_PLAN_GOAL_OUTSIDE_MAP', 0, start_time)
        if not self.is_free(start_xy[0], start_xy[1]):
            return self._result('NO_PLAN_START_IN_OBSTACLE', 0, start_time)
        if not self.is_free(goal_xy[0], goal_xy[1]):
            return self._result('NO_PLAN_GOAL_IN_OBSTACLE', 0, start_time)

        self.nodes.append(RRTNode(start_xy[0], start_xy[1], None, 0.0))

        for iteration in range(1, self.max_iterations + 1):
            if time.monotonic() - start_time > self.planning_timeout:
                return self._result(
                    'NO_PLAN_MAX_ITERATIONS', iteration, start_time)

            sample = self.goal_xy if random.random() < self.goal_sample_rate \
                else self.sample_free()
            nearest_index = self.nearest(sample)
            nearest_node = self.nodes[nearest_index]
            new_node = self.steer(nearest_node, sample)

            if not self.is_free(new_node.x, new_node.y):
                continue
            if not self.collision_free(
                    (nearest_node.x, nearest_node.y), (new_node.x, new_node.y)):
                continue

            near_indices = self.near(new_node)
            new_node = self.choose_parent(new_node, near_indices)
            self.nodes.append(new_node)
            new_index = len(self.nodes) - 1
            self.rewire(new_index, near_indices)

            goal_node = self._try_connect_goal(new_index)
            if goal_node is not None:
                self.nodes.append(goal_node)
                goal_index = len(self.nodes) - 1
                path = self.extract_path(goal_index)
                return self._result('SUCCESS', iteration, start_time, path)

        return self._result(
            'NO_PLAN_MAX_ITERATIONS', self.max_iterations, start_time)

    def sample_free(self) -> Tuple[float, float]:
        if self.costmap is None:
            raise RuntimeError('Costmap has not been set')
        for _ in range(100):
            x = random.uniform(self.costmap.x_min, self.costmap.x_max)
            y = random.uniform(self.costmap.y_min, self.costmap.y_max)
            if self.is_free(x, y):
                return x, y
        return (
            random.uniform(self.costmap.x_min, self.costmap.x_max),
            random.uniform(self.costmap.y_min, self.costmap.y_max),
        )

    def nearest(self, sample: Tuple[float, float]) -> int:
        return min(
            range(len(self.nodes)),
            key=lambda i: self._distance_xy(
                (self.nodes[i].x, self.nodes[i].y), sample),
        )

    def steer(self, from_node: RRTNode, to_point: Tuple[float, float]) -> RRTNode:
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        distance = math.hypot(dx, dy)
        if distance <= self.step_size:
            new_x, new_y = to_point
        else:
            scale = self.step_size / distance
            new_x = from_node.x + dx * scale
            new_y = from_node.y + dy * scale
        return RRTNode(new_x, new_y, None, math.inf)

    def collision_free(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> bool:
        distance = self._distance_xy(p1, p2)
        steps = max(1, int(math.ceil(
            distance / max(self.collision_check_resolution, 1e-6))))
        for step in range(steps + 1):
            t = step / steps
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            if not self.is_free(x, y):
                return False
        return True

    def is_free(self, x: float, y: float) -> bool:
        if self.costmap is None:
            raise RuntimeError('Costmap has not been set')
        col, row = self.world_to_grid(x, y)
        if not self._is_inside_grid(col, row):
            return False
        return self.costmap.data[self.costmap.index(
            col, row)] < self.occupied_threshold

    def near(self, new_node: RRTNode) -> List[int]:
        count = max(len(self.nodes), 2)
        adaptive_radius = self.search_radius * math.sqrt(math.log(count) / count)
        radius = max(self.step_size * 1.5, min(self.search_radius, adaptive_radius))
        return [
            index for index, node in enumerate(self.nodes)
            if self._distance_nodes(node, new_node) <= radius
        ]

    def choose_parent(
        self,
        new_node: RRTNode,
        near_indices: Sequence[int],
    ) -> RRTNode:
        best_parent = self.nearest((new_node.x, new_node.y))
        best_cost = self.nodes[best_parent].cost + self._distance_nodes(
            self.nodes[best_parent], new_node)

        for index in near_indices:
            near_node = self.nodes[index]
            segment_cost = self._distance_nodes(near_node, new_node)
            candidate_cost = near_node.cost + segment_cost
            if candidate_cost >= best_cost:
                continue
            if not self.collision_free(
                    (near_node.x, near_node.y), (new_node.x, new_node.y)):
                continue
            best_parent = index
            best_cost = candidate_cost

        new_node.parent = best_parent
        new_node.cost = best_cost
        return new_node

    def rewire(self, new_index: int, near_indices: Sequence[int]) -> None:
        new_node = self.nodes[new_index]
        for index in near_indices:
            if index == new_node.parent:
                continue
            near_node = self.nodes[index]
            candidate_cost = new_node.cost + self._distance_nodes(
                new_node, near_node)
            if candidate_cost >= near_node.cost:
                continue
            if not self.collision_free(
                    (new_node.x, new_node.y), (near_node.x, near_node.y)):
                continue
            near_node.parent = new_index
            near_node.cost = candidate_cost

    def extract_path(self, goal_node_index: int) -> List[Tuple[float, float]]:
        path = []
        current = goal_node_index
        while current is not None:
            node = self.nodes[current]
            path.append((node.x, node.y))
            current = node.parent
        path.reverse()
        return path

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        if self.costmap is None:
            raise RuntimeError('Costmap has not been set')
        col = int((x - self.costmap.origin[0]) / self.costmap.resolution)
        row = int((y - self.costmap.origin[1]) / self.costmap.resolution)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        if self.costmap is None:
            raise RuntimeError('Costmap has not been set')
        x = self.costmap.origin[0] + (col + 0.5) * self.costmap.resolution
        y = self.costmap.origin[1] + (row + 0.5) * self.costmap.resolution
        return x, y

    def _try_connect_goal(self, new_index: int) -> Optional[RRTNode]:
        new_node = self.nodes[new_index]
        distance_to_goal = self._distance_xy(
            (new_node.x, new_node.y), self.goal_xy)
        if distance_to_goal > self.goal_tolerance:
            return None
        if not self.collision_free((new_node.x, new_node.y), self.goal_xy):
            return None
        return RRTNode(
            self.goal_xy[0],
            self.goal_xy[1],
            new_index,
            new_node.cost + distance_to_goal,
        )

    def _result(
        self,
        status: str,
        iterations: int,
        start_time: float,
        path: Optional[List[Tuple[float, float]]] = None,
    ) -> RRTStarResult:
        return RRTStarResult(
            status=status,
            path=path or [],
            nodes=list(self.nodes),
            edges=self._edges(),
            iterations=iterations,
            elapsed_time=time.monotonic() - start_time,
        )

    def _edges(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        edges = []
        for node in self.nodes:
            if node.parent is None:
                continue
            parent = self.nodes[node.parent]
            edges.append(((parent.x, parent.y), (node.x, node.y)))
        return edges

    def _is_inside_world(self, x: float, y: float) -> bool:
        if self.costmap is None:
            raise RuntimeError('Costmap has not been set')
        return self.costmap.x_min <= x < self.costmap.x_max and \
            self.costmap.y_min <= y < self.costmap.y_max

    def _is_inside_grid(self, col: int, row: int) -> bool:
        if self.costmap is None:
            raise RuntimeError('Costmap has not been set')
        return 0 <= col < self.costmap.width and 0 <= row < self.costmap.height

    @staticmethod
    def _distance_nodes(node_a: RRTNode, node_b: RRTNode) -> float:
        return math.hypot(node_a.x - node_b.x, node_a.y - node_b.y)

    @staticmethod
    def _distance_xy(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
    ) -> float:
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
