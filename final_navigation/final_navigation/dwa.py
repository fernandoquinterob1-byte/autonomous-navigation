from dataclasses import dataclass, field
import math
from typing import List, Optional, Sequence, Tuple

from final_navigation.rrt_star import PlanningCostmap


@dataclass(frozen=True)
class DWAConfig:
    prediction_time: float = 2.0
    dt: float = 0.1
    v_min: float = 0.0
    v_max: float = 0.35
    w_min: float = -1.2
    w_max: float = 1.2
    v_samples: int = 6
    w_samples: int = 13
    lookahead_distance: float = 0.8
    collision_distance: float = 0.22
    safety_distance: float = 0.45
    occupied_threshold: int = 50
    goal_weight: float = 2.0
    path_weight: float = 1.0
    obstacle_weight: float = 3.0
    heading_weight: float = 1.0
    velocity_weight: float = 0.5
    turn_switch_penalty: float = 2.0
    rotation_only_penalty: float = 1.0
    use_costmap_collision: bool = True
    use_scan_collision: bool = True


@dataclass
class TrajectoryScore:
    trajectory: List[Tuple[float, float, float]]
    v: float
    w: float
    score: float
    collision: bool = False
    min_obstacle_distance: float = math.inf


@dataclass
class DWAResult:
    status: str
    v: float = 0.0
    w: float = 0.0
    local_goal: Optional[Tuple[float, float]] = None
    selected_trajectory: List[Tuple[float, float, float]] = field(
        default_factory=list)
    candidate_trajectories: List[List[Tuple[float, float, float]]] = field(
        default_factory=list)


class DWAController:
    def __init__(self, config: Optional[DWAConfig] = None):
        self.config = config or DWAConfig()

    def compute_command(
        self,
        pose: Tuple[float, float, float],
        global_path: Sequence[Tuple[float, float]],
        scan=None,
        costmap: Optional[PlanningCostmap] = None,
        current_v: float = 0.0,
        current_w: float = 0.0,
    ) -> DWAResult:
        if not global_path:
            return DWAResult(status='WAITING_FOR_PATH')

        local_goal = self.select_local_goal(pose, global_path)
        scan_points = self.scan_to_points(scan, pose) if scan is not None else []
        candidates = []
        best: Optional[TrajectoryScore] = None

        for v, w in self.sample_velocities(current_v, current_w):
            if abs(v) < 1e-6 and abs(w) < 1e-6 and \
                    self._distance_xy((pose[0], pose[1]), local_goal) > 0.05:
                continue
            trajectory = self.simulate_trajectory(pose, v, w)
            candidates.append(trajectory)
            score = self.score_trajectory(
                trajectory, local_goal, global_path, scan_points, costmap, v, w)
            if score.collision:
                continue
            if abs(current_w) > 1e-3 and current_w * w < 0.0 and \
                    current_v <= 0.05:
                score.score += self.config.turn_switch_penalty
            if v <= 1e-3 and abs(w) > 1e-3:
                score.score += self.config.rotation_only_penalty
            if best is None or score.score < best.score:
                best = score

        if best is None:
            return DWAResult(
                status='NO_VALID_TRAJECTORY',
                local_goal=local_goal,
                candidate_trajectories=candidates,
            )

        return DWAResult(
            status='FOLLOWING_PATH',
            v=best.v,
            w=best.w,
            local_goal=local_goal,
            selected_trajectory=best.trajectory,
            candidate_trajectories=candidates,
        )

    def select_local_goal(
        self,
        pose: Tuple[float, float, float],
        global_path: Sequence[Tuple[float, float]],
    ) -> Tuple[float, float]:
        robot_xy = (pose[0], pose[1])
        closest_index = min(
            range(len(global_path)),
            key=lambda i: self._distance_xy(robot_xy, global_path[i]),
        )

        accumulated = 0.0
        previous = global_path[closest_index]
        for index in range(closest_index + 1, len(global_path)):
            current = global_path[index]
            accumulated += self._distance_xy(previous, current)
            if accumulated >= self.config.lookahead_distance:
                return current
            previous = current
        return global_path[-1]

    def sample_velocities(
        self,
        current_v: float = 0.0,
        current_w: float = 0.0,
    ) -> List[Tuple[float, float]]:
        del current_v, current_w
        v_values = self._linspace(
            self.config.v_min, self.config.v_max, self.config.v_samples)
        w_values = self._linspace(
            self.config.w_min, self.config.w_max, self.config.w_samples)
        return [(v, w) for v in v_values for w in w_values]

    def simulate_trajectory(
        self,
        pose: Tuple[float, float, float],
        v: float,
        w: float,
    ) -> List[Tuple[float, float, float]]:
        x, y, theta = pose
        trajectory = [(x, y, theta)]
        steps = max(1, int(math.ceil(
            self.config.prediction_time / max(self.config.dt, 1e-6))))
        for _ in range(steps):
            x += v * math.cos(theta) * self.config.dt
            y += v * math.sin(theta) * self.config.dt
            theta = self.normalize_angle(theta + w * self.config.dt)
            trajectory.append((x, y, theta))
        return trajectory

    def score_trajectory(
        self,
        trajectory: Sequence[Tuple[float, float, float]],
        local_goal: Tuple[float, float],
        global_path: Sequence[Tuple[float, float]],
        scan_points: Sequence[Tuple[float, float]],
        costmap: Optional[PlanningCostmap],
        v: float = 0.0,
        w: float = 0.0,
    ) -> TrajectoryScore:
        if self.config.use_costmap_collision and costmap is not None:
            if self.check_collision_costmap(trajectory, costmap):
                return TrajectoryScore(list(trajectory), v, w, math.inf, True)

        min_scan_distance = math.inf
        if self.config.use_scan_collision and scan_points:
            collision, min_scan_distance = self.check_collision_scan(
                trajectory, scan_points)
            if collision:
                return TrajectoryScore(
                    list(trajectory), v, w, math.inf, True,
                    min_scan_distance)

        end = trajectory[-1]
        end_xy = (end[0], end[1])
        goal_cost = self.distance_to_goal(end_xy, local_goal)
        path_cost = self.distance_to_path(end_xy, global_path)
        heading_target = math.atan2(
            local_goal[1] - end[1], local_goal[0] - end[0])
        heading_cost = abs(self.normalize_angle(heading_target - end[2]))
        velocity_cost = max(0.0, self.config.v_max - v)
        obstacle_cost = self._obstacle_cost(min_scan_distance)

        score = (
            self.config.goal_weight * goal_cost +
            self.config.path_weight * path_cost +
            self.config.obstacle_weight * obstacle_cost +
            self.config.heading_weight * heading_cost +
            self.config.velocity_weight * velocity_cost
        )
        return TrajectoryScore(
            list(trajectory), v, w, score, False, min_scan_distance)

    def check_collision_costmap(
        self,
        trajectory: Sequence[Tuple[float, float, float]],
        costmap: PlanningCostmap,
    ) -> bool:
        for x, y, _ in trajectory:
            col, row = self.world_to_grid(x, y, costmap)
            if not self._is_inside_grid(col, row, costmap):
                return True
            value = costmap.data[costmap.index(col, row)]
            if value >= self.config.occupied_threshold:
                return True
        return False

    def check_collision_scan(
        self,
        trajectory: Sequence[Tuple[float, float, float]],
        scan_points: Sequence[Tuple[float, float]],
    ) -> Tuple[bool, float]:
        min_distance = self._min_distance_to_points(trajectory, scan_points)
        return min_distance < self.config.collision_distance, min_distance

    def scan_to_points(
        self,
        scan,
        robot_pose: Tuple[float, float, float],
    ) -> List[Tuple[float, float]]:
        if scan is None:
            return []
        points = []
        angle = scan.angle_min
        robot_x, robot_y, robot_theta = robot_pose
        for range_value in scan.ranges:
            if math.isfinite(range_value) and \
                    scan.range_min <= range_value <= scan.range_max:
                global_angle = robot_theta + angle
                points.append((
                    robot_x + range_value * math.cos(global_angle),
                    robot_y + range_value * math.sin(global_angle),
                ))
            angle += scan.angle_increment
        return points

    def distance_to_path(
        self,
        point: Tuple[float, float],
        path: Sequence[Tuple[float, float]],
    ) -> float:
        if not path:
            return math.inf
        if len(path) == 1:
            return self._distance_xy(point, path[0])
        return min(
            self._point_to_segment_distance(point, path[i], path[i + 1])
            for i in range(len(path) - 1)
        )

    def distance_to_goal(
        self,
        point: Tuple[float, float],
        local_goal: Tuple[float, float],
    ) -> float:
        return self._distance_xy(point, local_goal)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def world_to_grid(
        x: float,
        y: float,
        costmap: PlanningCostmap,
    ) -> Tuple[int, int]:
        col = int((x - costmap.origin[0]) / costmap.resolution)
        row = int((y - costmap.origin[1]) / costmap.resolution)
        return col, row

    def _obstacle_cost(self, min_distance: float) -> float:
        if not math.isfinite(min_distance):
            return 0.0
        if min_distance >= self.config.safety_distance:
            return 0.0
        span = max(self.config.safety_distance - self.config.collision_distance,
                   1e-6)
        return (self.config.safety_distance - min_distance) / span

    def _min_distance_to_points(
        self,
        trajectory: Sequence[Tuple[float, float, float]],
        points: Sequence[Tuple[float, float]],
    ) -> float:
        if not points:
            return math.inf
        min_distance = math.inf
        for x, y, _ in trajectory:
            for point in points:
                min_distance = min(min_distance, self._distance_xy((x, y), point))
        return min_distance

    @staticmethod
    def _point_to_segment_distance(
        point: Tuple[float, float],
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> float:
        px, py = point
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            return math.hypot(px - sx, py - sy)
        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) /
                         length_squared))
        projection = (sx + t * dx, sy + t * dy)
        return math.hypot(px - projection[0], py - projection[1])

    @staticmethod
    def _distance_xy(
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
    ) -> float:
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    @staticmethod
    def _is_inside_grid(col: int, row: int, costmap: PlanningCostmap) -> bool:
        return 0 <= col < costmap.width and 0 <= row < costmap.height

    @staticmethod
    def _linspace(start: float, stop: float, count: int) -> List[float]:
        if count <= 1:
            return [(start + stop) * 0.5]
        step = (stop - start) / (count - 1)
        return [start + i * step for i in range(count)]
