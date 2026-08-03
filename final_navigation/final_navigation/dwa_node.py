import math
from typing import List, Optional, Tuple

from final_navigation.dwa import DWAConfig, DWAController
from final_navigation.rrt_star import PlanningCostmap
from final_navigation.utils import yaw_to_quaternion
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker


class DWANode(Node):
    def __init__(self):
        super().__init__('dwa_node')
        self._declare_parameters()

        self.global_frame = self.get_parameter(
            'global_frame').get_parameter_value().string_value
        self.require_scan = self.get_parameter(
            'require_scan').get_parameter_value().bool_value
        self.publish_local_trajectories = self.get_parameter(
            'publish_local_trajectories').get_parameter_value().bool_value
        self.goal_tolerance_xy = self.get_parameter(
            'goal_tolerance_xy').get_parameter_value().double_value
        self.goal_tolerance_yaw = self.get_parameter(
            'goal_tolerance_yaw').get_parameter_value().double_value
        self.yaw_alignment_kp = self.get_parameter(
            'yaw_alignment_kp').get_parameter_value().double_value
        self.yaw_alignment_min_w = self.get_parameter(
            'yaw_alignment_min_w').get_parameter_value().double_value
        self.yaw_alignment_max_w = self.get_parameter(
            'yaw_alignment_max_w').get_parameter_value().double_value
        self.control_frequency = self.get_parameter(
            'control_frequency').get_parameter_value().double_value
        self.recovery_enabled = self.get_parameter(
            'recovery_enabled').get_parameter_value().bool_value
        self.recovery_stuck_timeout = self.get_parameter(
            'recovery_stuck_timeout').get_parameter_value().double_value
        self.recovery_progress_distance = self.get_parameter(
            'recovery_progress_distance').get_parameter_value().double_value
        self.recovery_trigger_distance = self.get_parameter(
            'recovery_trigger_distance').get_parameter_value().double_value
        self.recovery_front_clearance = self.get_parameter(
            'recovery_front_clearance').get_parameter_value().double_value
        self.recovery_follow_distance = self.get_parameter(
            'recovery_follow_distance').get_parameter_value().double_value
        self.recovery_forward_speed = self.get_parameter(
            'recovery_forward_speed').get_parameter_value().double_value
        self.recovery_turn_speed = self.get_parameter(
            'recovery_turn_speed').get_parameter_value().double_value
        self.recovery_side_kp = self.get_parameter(
            'recovery_side_kp').get_parameter_value().double_value
        self.recovery_max_w = self.get_parameter(
            'recovery_max_w').get_parameter_value().double_value
        self.recovery_min_travel = self.get_parameter(
            'recovery_min_travel').get_parameter_value().double_value
        self.recovery_switch_timeout = self.get_parameter(
            'recovery_switch_timeout').get_parameter_value().double_value
        self.recovery_turn_timeout = self.get_parameter(
            'recovery_turn_timeout').get_parameter_value().double_value
        self.recovery_timeout = self.get_parameter(
            'recovery_timeout').get_parameter_value().double_value
        self.recovery_prediction_time = self.get_parameter(
            'recovery_prediction_time').get_parameter_value().double_value

        self.controller = DWAController(DWAConfig(
            prediction_time=self.get_parameter(
                'prediction_time').get_parameter_value().double_value,
            dt=self.get_parameter('dt').get_parameter_value().double_value,
            v_min=self.get_parameter('v_min').get_parameter_value().double_value,
            v_max=self.get_parameter('v_max').get_parameter_value().double_value,
            w_min=self.get_parameter('w_min').get_parameter_value().double_value,
            w_max=self.get_parameter('w_max').get_parameter_value().double_value,
            v_samples=self.get_parameter(
                'v_samples').get_parameter_value().integer_value,
            w_samples=self.get_parameter(
                'w_samples').get_parameter_value().integer_value,
            lookahead_distance=self.get_parameter(
                'lookahead_distance').get_parameter_value().double_value,
            collision_distance=self.get_parameter(
                'collision_distance').get_parameter_value().double_value,
            safety_distance=self.get_parameter(
                'safety_distance').get_parameter_value().double_value,
            occupied_threshold=self.get_parameter(
                'occupied_threshold').get_parameter_value().integer_value,
            goal_weight=self.get_parameter(
                'goal_weight').get_parameter_value().double_value,
            path_weight=self.get_parameter(
                'path_weight').get_parameter_value().double_value,
            obstacle_weight=self.get_parameter(
                'obstacle_weight').get_parameter_value().double_value,
            heading_weight=self.get_parameter(
                'heading_weight').get_parameter_value().double_value,
            velocity_weight=self.get_parameter(
                'velocity_weight').get_parameter_value().double_value,
            turn_switch_penalty=self.get_parameter(
                'turn_switch_penalty').get_parameter_value().double_value,
            rotation_only_penalty=self.get_parameter(
                'rotation_only_penalty').get_parameter_value().double_value,
            use_costmap_collision=self.get_parameter(
                'use_costmap_collision').get_parameter_value().bool_value,
            use_scan_collision=self.get_parameter(
                'use_scan_collision').get_parameter_value().bool_value,
        ))

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.path_sub = self.create_subscription(
            Path, '/final/global_path', self._path_callback, transient_qos)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/final/goal_pose', self._goal_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/ground_truth_pose', self._pose_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/final/inflated_costmap', self._costmap_callback,
            transient_qos)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(
            String, '/final/controller_status', transient_qos)
        self.local_traj_pub = self.create_publisher(
            Marker, '/final/local_trajectories', transient_qos)
        self.selected_traj_pub = self.create_publisher(
            Marker, '/final/selected_trajectory', transient_qos)
        self.local_goal_pub = self.create_publisher(
            PoseStamped, '/final/local_goal', transient_qos)

        self.global_path: List[Tuple[float, float]] = []
        self.final_goal_yaw: Optional[float] = None
        self.pose: Optional[Tuple[float, float, float]] = None
        self.scan: Optional[LaserScan] = None
        self.costmap: Optional[PlanningCostmap] = None
        self.last_status: Optional[str] = None
        self.current_v = 0.0
        self.current_w = 0.0
        self.progress_pose: Optional[Tuple[float, float]] = None
        self.progress_time: Optional[float] = None
        self.recovery_active = False
        self.recovery_direction = 0
        self.recovery_phase = 'IDLE'
        self.recovery_phase_start_time: Optional[float] = None
        self.recovery_start_time: Optional[float] = None
        self.recovery_progress_pose: Optional[Tuple[float, float]] = None
        self.recovery_progress_time: Optional[float] = None
        self.recovery_travel_distance = 0.0

        period = 1.0 / max(self.control_frequency, 0.001)
        self.timer = self.create_timer(period, self._control_loop)
        self._publish_status('WAITING_FOR_PATH')
        self.get_logger().info(
            f'DWA controller running at {self.control_frequency:.1f} Hz')

    def _declare_parameters(self):
        self.declare_parameter('enabled', True)
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('control_frequency', 10.0)
        self.declare_parameter('prediction_time', 2.0)
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('v_min', 0.0)
        self.declare_parameter('v_max', 0.35)
        self.declare_parameter('w_min', -1.2)
        self.declare_parameter('w_max', 1.2)
        self.declare_parameter('v_samples', 6)
        self.declare_parameter('w_samples', 13)
        self.declare_parameter('lookahead_distance', 0.8)
        self.declare_parameter('goal_tolerance_xy', 0.25)
        self.declare_parameter('goal_tolerance_yaw', 0.30)
        self.declare_parameter('yaw_alignment_kp', 0.8)
        self.declare_parameter('yaw_alignment_min_w', 0.04)
        self.declare_parameter('yaw_alignment_max_w', 0.35)
        self.declare_parameter('collision_distance', 0.22)
        self.declare_parameter('safety_distance', 0.45)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('goal_weight', 2.0)
        self.declare_parameter('path_weight', 1.0)
        self.declare_parameter('obstacle_weight', 3.0)
        self.declare_parameter('heading_weight', 1.0)
        self.declare_parameter('velocity_weight', 0.5)
        self.declare_parameter('turn_switch_penalty', 2.0)
        self.declare_parameter('rotation_only_penalty', 1.0)
        self.declare_parameter('recovery_enabled', True)
        self.declare_parameter('recovery_stuck_timeout', 2.0)
        self.declare_parameter('recovery_progress_distance', 0.08)
        self.declare_parameter('recovery_trigger_distance', 0.90)
        self.declare_parameter('recovery_front_clearance', 0.55)
        self.declare_parameter('recovery_follow_distance', 0.45)
        self.declare_parameter('recovery_forward_speed', 0.15)
        self.declare_parameter('recovery_turn_speed', 0.32)
        self.declare_parameter('recovery_side_kp', 1.2)
        self.declare_parameter('recovery_max_w', 0.40)
        self.declare_parameter('recovery_min_travel', 0.60)
        self.declare_parameter('recovery_switch_timeout', 8.0)
        self.declare_parameter('recovery_turn_timeout', 12.0)
        self.declare_parameter('recovery_timeout', 180.0)
        self.declare_parameter('recovery_prediction_time', 1.0)
        self.declare_parameter('require_scan', False)
        self.declare_parameter('use_costmap_collision', True)
        self.declare_parameter('use_scan_collision', True)
        self.declare_parameter('publish_local_trajectories', True)

    def _path_callback(self, msg: Path):
        self.global_path = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]
        self.final_goal_yaw = None
        if msg.poses:
            self.final_goal_yaw = self._yaw_from_quaternion(
                msg.poses[-1].pose.orientation)
        self._reset_recovery_tracking()
        if self.global_path:
            self.get_logger().info(
                f'Received global path with {len(self.global_path)} points')
        else:
            self.get_logger().warn('Received empty global path')

    def _goal_callback(self, msg: PoseStamped):
        del msg
        self.global_path = []
        self.final_goal_yaw = None
        self._reset_recovery_tracking()
        self._stop('WAITING_FOR_PATH')

    def _pose_callback(self, msg: PoseStamped):
        if msg.header.frame_id and msg.header.frame_id not in (
                self.global_frame, 'world_demo'):
            self.get_logger().warn(
                f'Pose frame is {msg.header.frame_id}; assuming coordinates '
                f'are compatible with {self.global_frame}')
        self.pose = (
            msg.pose.position.x,
            msg.pose.position.y,
            self._yaw_from_quaternion(msg.pose.orientation),
        )

    def _scan_callback(self, msg: LaserScan):
        self.scan = msg

    def _costmap_callback(self, msg: OccupancyGrid):
        self.costmap = PlanningCostmap(
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin=(
                msg.info.origin.position.x,
                msg.info.origin.position.y,
                self._yaw_from_quaternion(msg.info.origin.orientation),
            ),
            data=list(msg.data),
            occupied_threshold=self.controller.config.occupied_threshold,
        )

    def _control_loop(self):
        if not self.global_path:
            self._stop('WAITING_FOR_PATH')
            return
        if self.pose is None:
            self._stop('WAITING_FOR_POSE')
            return
        if self.require_scan and self.scan is None:
            self._stop('WAITING_FOR_SCAN')
            return

        final_goal = self.global_path[-1]
        if self._distance_xy((self.pose[0], self.pose[1]), final_goal) <= \
                self.goal_tolerance_xy:
            self._finish_recovery()
            if self.final_goal_yaw is not None:
                yaw_error = self._normalize_angle(
                    self.final_goal_yaw - self.pose[2])
                if abs(yaw_error) > self.goal_tolerance_yaw:
                    self._publish_cmd(0.0, self._alignment_velocity(yaw_error))
                    self._publish_status('ALIGNING_GOAL_YAW')
                    self._publish_local_goal(final_goal)
                    self._publish_selected_trajectory([])
                    return
            self._stop('GOAL_REACHED')
            return

        local_goal = self.controller.select_local_goal(
            self.pose, self.global_path)
        now_sec = self._now_sec()
        self._update_progress(now_sec)

        if self.recovery_active:
            if self._run_recovery(local_goal, now_sec):
                return
        elif self._should_start_recovery(now_sec):
            self._start_recovery(now_sec)
            if self._run_recovery(local_goal, now_sec):
                return

        result = self.controller.compute_command(
            self.pose,
            self.global_path,
            scan=self.scan,
            costmap=self.costmap,
            current_v=self.current_v,
            current_w=self.current_w,
        )

        if result.status == 'FOLLOWING_PATH':
            self._publish_cmd(result.v, result.w)
            self._publish_status('FOLLOWING_PATH')
        else:
            self._stop(result.status)

        self._publish_local_goal(result.local_goal)
        self._publish_local_trajectories(result.candidate_trajectories)
        self._publish_selected_trajectory(result.selected_trajectory)

    def _reset_recovery_tracking(self):
        self.recovery_active = False
        self.recovery_direction = 0
        self.recovery_phase = 'IDLE'
        self.recovery_phase_start_time = None
        self.recovery_start_time = None
        self.recovery_progress_pose = None
        self.recovery_progress_time = None
        self.recovery_travel_distance = 0.0
        if self.pose is None:
            self.progress_pose = None
            self.progress_time = None
            return
        self.progress_pose = (self.pose[0], self.pose[1])
        self.progress_time = self._now_sec()

    def _update_progress(self, now_sec: float):
        current = (self.pose[0], self.pose[1])
        if self.progress_pose is None or self.progress_time is None:
            self.progress_pose = current
            self.progress_time = now_sec
            return
        if self._distance_xy(current, self.progress_pose) >= \
                self.recovery_progress_distance:
            self.progress_pose = current
            self.progress_time = now_sec

    def _should_start_recovery(self, now_sec: float) -> bool:
        if not self.recovery_enabled or self.scan is None:
            return False
        if self.progress_time is None:
            return False
        if now_sec - self.progress_time < self.recovery_stuck_timeout:
            return False
        forward_clearance = self._sector_clearance(
            0.0, math.radians(50.0))
        return forward_clearance < self.recovery_trigger_distance

    def _start_recovery(self, now_sec: float):
        left_clearance = self._sector_clearance(
            math.pi * 0.5, math.radians(50.0))
        right_clearance = self._sector_clearance(
            -math.pi * 0.5, math.radians(50.0))
        if abs(left_clearance - right_clearance) <= 0.05 and \
                abs(self.current_w) > 1e-3:
            direction = 1 if self.current_w > 0.0 else -1
        else:
            direction = 1 if left_clearance >= right_clearance else -1

        current = (self.pose[0], self.pose[1])
        self.recovery_active = True
        self.recovery_direction = direction
        self.recovery_phase = 'TURNING_AROUND_CORNER'
        self.recovery_phase_start_time = now_sec
        self.recovery_start_time = now_sec
        self.recovery_progress_pose = current
        self.recovery_progress_time = now_sec
        self.recovery_travel_distance = 0.0
        side = 'left' if direction > 0 else 'right'
        self.get_logger().warn(
            f'Obstacle recovery started: following {side} side '
            f'(left={left_clearance:.2f} m, '
            f'right={right_clearance:.2f} m)')

    def _finish_recovery(self):
        if not self.recovery_active:
            return
        self.get_logger().info(
            'Obstacle recovery finished; returning control to DWA')
        self.recovery_active = False
        self.recovery_direction = 0
        self.recovery_phase = 'IDLE'
        self.recovery_phase_start_time = None
        self.recovery_start_time = None
        self.recovery_progress_pose = None
        self.recovery_progress_time = None
        self.recovery_travel_distance = 0.0
        if self.pose is not None:
            self.progress_pose = (self.pose[0], self.pose[1])
            self.progress_time = self._now_sec()

    def _run_recovery(
        self,
        local_goal: Tuple[float, float],
        now_sec: float,
    ) -> bool:
        if self.recovery_start_time is None:
            return False
        if now_sec - self.recovery_start_time >= self.recovery_timeout:
            self.get_logger().error(
                'Obstacle recovery timed out: no safe bypass found')
            self._stop('NO_VALID_TRAJECTORY')
            self._publish_local_goal(local_goal)
            self._publish_selected_trajectory([])
            return True

        current = (self.pose[0], self.pose[1])
        movement = 0.0
        if self.recovery_progress_pose is None or \
                self.recovery_progress_time is None:
            self.recovery_progress_pose = current
            self.recovery_progress_time = now_sec
        else:
            movement = self._distance_xy(
                current, self.recovery_progress_pose)
        if movement >= self.recovery_progress_distance:
            self.recovery_travel_distance += movement
            self.recovery_progress_pose = current
            self.recovery_progress_time = now_sec

        if self.recovery_travel_distance >= self.recovery_min_travel and \
                self._goal_direction_is_clear(local_goal):
            self._finish_recovery()
            return False

        front_clearance = self._sector_clearance(
            0.0, math.radians(22.5))
        follow_trajectory = None
        if front_clearance >= self.recovery_front_clearance:
            followed_side = -self.recovery_direction * math.pi * 0.5
            side_clearance = self._sector_clearance(
                followed_side, math.radians(30.0))
            v = min(self.recovery_forward_speed,
                    self.controller.config.v_max)
            if math.isfinite(side_clearance) and \
                    side_clearance < 0.95 * self.scan.range_max:
                error = self.recovery_follow_distance - side_clearance
                w = self.recovery_direction * self.recovery_side_kp * error
                w = self._clamp(
                    w, -self.recovery_max_w, self.recovery_max_w)
            else:
                w = 0.0
            follow_trajectory = self._simulate_recovery_trajectory(v, w)

        can_follow = follow_trajectory is not None and \
            self._trajectory_is_safe(follow_trajectory)
        if can_follow:
            if self.recovery_phase != 'FOLLOWING_EDGE':
                self.recovery_phase = 'FOLLOWING_EDGE'
                self.recovery_phase_start_time = now_sec
                self.recovery_progress_pose = current
                self.recovery_progress_time = now_sec
                self.get_logger().info(
                    'Obstacle recovery phase: FOLLOWING_EDGE')
            elif self.recovery_progress_time is not None and \
                    now_sec - self.recovery_progress_time >= \
                    self.recovery_switch_timeout:
                self._switch_recovery_side(
                    current, now_sec, 'no linear progress while following edge')
                can_follow = False

        if not can_follow:
            if self.recovery_phase != 'TURNING_AROUND_CORNER':
                self.recovery_phase = 'TURNING_AROUND_CORNER'
                self.recovery_phase_start_time = now_sec
                self.get_logger().info(
                    'Obstacle recovery phase: TURNING_AROUND_CORNER')
            elif self.recovery_phase_start_time is not None and \
                    now_sec - self.recovery_phase_start_time >= \
                    self.recovery_turn_timeout:
                self._switch_recovery_side(
                    current, now_sec, 'turning phase timed out')
            v = 0.0
            w = self.recovery_direction * self.recovery_turn_speed
            trajectory = self._simulate_recovery_trajectory(v, w)
        else:
            trajectory = follow_trajectory

        self._publish_cmd(v, w)
        self._publish_status('AVOIDING_OBSTACLE')
        self._publish_local_goal(local_goal)
        self._publish_local_trajectories([trajectory])
        self._publish_selected_trajectory(trajectory)
        return True

    def _switch_recovery_side(
        self,
        current: Tuple[float, float],
        now_sec: float,
        reason: str,
    ):
        self.recovery_direction *= -1
        self.recovery_phase = 'TURNING_AROUND_CORNER'
        self.recovery_phase_start_time = now_sec
        self.recovery_progress_pose = current
        self.recovery_progress_time = now_sec
        self.recovery_travel_distance = 0.0
        side = 'left' if self.recovery_direction > 0 else 'right'
        self.get_logger().warn(
            f'Obstacle recovery switching to {side}: {reason}')

    def _goal_direction_is_clear(
        self,
        local_goal: Tuple[float, float],
    ) -> bool:
        dx = local_goal[0] - self.pose[0]
        dy = local_goal[1] - self.pose[1]
        distance = math.hypot(dx, dy)
        if distance <= self.goal_tolerance_xy:
            return True

        relative_bearing = self._normalize_angle(
            math.atan2(dy, dx) - self.pose[2])
        scan_clearance = self._sector_clearance(
            relative_bearing, math.radians(8.0))
        required_clearance = min(
            distance, self.controller.config.lookahead_distance)
        required_clearance += self.controller.config.collision_distance
        if scan_clearance < required_clearance:
            return False
        return self._costmap_line_is_free(
            (self.pose[0], self.pose[1]), local_goal)

    def _costmap_line_is_free(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> bool:
        if not self.controller.config.use_costmap_collision or \
                self.costmap is None:
            return True
        distance = self._distance_xy(start, end)
        step = max(self.costmap.resolution, 0.02)
        count = max(1, int(math.ceil(distance / step)))
        trajectory = []
        for index in range(count + 1):
            ratio = index / count
            trajectory.append((
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
                0.0,
            ))
        return not self.controller.check_collision_costmap(
            trajectory, self.costmap)

    def _trajectory_is_safe(self, trajectory) -> bool:
        if self.controller.config.use_costmap_collision and \
                self.costmap is not None and \
                self.controller.check_collision_costmap(
                    trajectory, self.costmap):
            return False
        if self.controller.config.use_scan_collision and self.scan is not None:
            scan_points = self.controller.scan_to_points(
                self.scan, self.pose)
            collision, _ = self.controller.check_collision_scan(
                trajectory, scan_points)
            if collision:
                return False
        return True

    def _simulate_recovery_trajectory(
        self,
        v: float,
        w: float,
    ):
        x, y, theta = self.pose
        trajectory = [(x, y, theta)]
        dt = max(self.controller.config.dt, 1e-3)
        steps = max(1, int(math.ceil(
            self.recovery_prediction_time / dt)))
        for _ in range(steps):
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta = self._normalize_angle(theta + w * dt)
            trajectory.append((x, y, theta))
        return trajectory

    def _sector_clearance(
        self,
        center_angle: float,
        half_width: float,
    ) -> float:
        if self.scan is None:
            return math.inf
        clearance = math.inf
        angle = self.scan.angle_min
        for range_value in self.scan.ranges:
            difference = self._normalize_angle(angle - center_angle)
            if abs(difference) <= half_width:
                if math.isinf(range_value) and range_value > 0.0:
                    value = self.scan.range_max
                elif math.isfinite(range_value) and \
                        self.scan.range_min <= range_value <= \
                        self.scan.range_max:
                    value = range_value
                else:
                    angle += self.scan.angle_increment
                    continue
                clearance = min(clearance, value)
            angle += self.scan.angle_increment
        return clearance

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def _stop(self, status: str):
        self._publish_cmd(0.0, 0.0)
        self._publish_status(status)

    def _publish_cmd(self, v: float, w: float):
        self.current_v = float(v)
        self.current_w = float(w)
        msg = Twist()
        msg.linear.x = self.current_v
        msg.angular.z = self.current_w
        self.cmd_pub.publish(msg)

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        if status != self.last_status:
            self.get_logger().info(f'Controller status: {status}')
            self.last_status = status

    def _publish_local_goal(self, goal: Optional[Tuple[float, float]]):
        if goal is None:
            return
        msg = PoseStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(goal[0])
        msg.pose.position.y = float(goal[1])
        msg.pose.orientation.w = 1.0
        self.local_goal_pub.publish(msg)

    def _publish_local_trajectories(self, trajectories):
        if not self.publish_local_trajectories:
            return
        marker = self._base_marker('local_trajectories', 0, Marker.LINE_LIST)
        marker.scale.x = 0.015
        marker.color = ColorRGBA(r=0.4, g=0.4, b=0.4, a=0.35)
        for trajectory in trajectories:
            for i in range(len(trajectory) - 1):
                marker.points.append(self._point(trajectory[i], 0.03))
                marker.points.append(self._point(trajectory[i + 1], 0.03))
        self.local_traj_pub.publish(marker)

    def _publish_selected_trajectory(self, trajectory):
        marker = self._base_marker('selected_trajectory', 1, Marker.LINE_STRIP)
        marker.scale.x = 0.045
        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.2, a=0.9)
        for point in trajectory:
            marker.points.append(self._point(point, 0.06))
        self.selected_traj_pub.publish(marker)

    def _base_marker(self, namespace: str, marker_id: int, marker_type: int):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation = yaw_to_quaternion(0.0)
        return marker

    @staticmethod
    def _point(point, z: float):
        return Point(x=float(point[0]), y=float(point[1]), z=z)

    @staticmethod
    def _yaw_from_quaternion(quat) -> float:
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _alignment_velocity(self, yaw_error: float) -> float:
        physical_max_w = max(
            abs(self.controller.config.w_min),
            abs(self.controller.config.w_max),
        )
        max_w = min(abs(self.yaw_alignment_max_w), physical_max_w)
        min_w = min(abs(self.yaw_alignment_min_w), max_w)

        command = self.yaw_alignment_kp * yaw_error
        command = max(-max_w, min(max_w, command))

        if abs(yaw_error) <= self.goal_tolerance_yaw:
            return 0.0
        if 0.0 < abs(command) < min_w:
            command = math.copysign(min_w, yaw_error)
        return command

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def _distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


def main(args=None):
    rclpy.init(args=args)
    node = DWANode()
    try:
        rclpy.spin(node)
    finally:
        node._stop('STOPPED')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
