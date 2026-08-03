import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class NavigationManager(Node):
    def __init__(self):
        super().__init__('navigation_manager')
        self._declare_parameters()

        self.status_frequency = self.get_parameter(
            'status_frequency').get_parameter_value().double_value
        self.goal_tolerance_xy = self.get_parameter(
            'goal_tolerance_xy').get_parameter_value().double_value
        self.collision_distance = self.get_parameter(
            'collision_distance').get_parameter_value().double_value
        self.collision_risk_distance = self.get_parameter(
            'collision_risk_distance').get_parameter_value().double_value
        self.no_valid_trajectory_timeout = self.get_parameter(
            'no_valid_trajectory_timeout').get_parameter_value().double_value
        self.publish_stop_on_collision = self.get_parameter(
            'publish_stop_on_collision').get_parameter_value().bool_value

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, '/final/goal_pose', self._goal_callback, 10)
        self.planner_status_sub = self.create_subscription(
            String, '/final/planner_status', self._planner_status_callback, 10)
        self.controller_status_sub = self.create_subscription(
            String, '/final/controller_status',
            self._controller_status_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/ground_truth_pose', self._pose_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/final/global_path', self._path_callback, transient_qos)

        self.status_pub = self.create_publisher(
            String, '/final/navigation_status', transient_qos)
        self.event_pub = self.create_publisher(
            String, '/final/navigation_event', transient_qos)
        self.collision_status_pub = self.create_publisher(
            String, '/final/collision_status', transient_qos)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.goal_xy: Optional[Tuple[float, float]] = None
        self.pose_xy: Optional[Tuple[float, float]] = None
        self.has_path = False
        self.planner_status = 'IDLE'
        self.controller_status = 'IDLE'
        self.navigation_status = 'WAITING_FOR_GOAL'
        self.last_published_status: Optional[str] = None
        self.last_event: Optional[str] = None
        self.last_no_valid_time: Optional[float] = None
        self.min_scan_range = math.inf

        period = 1.0 / max(self.status_frequency, 0.001)
        self.timer = self.create_timer(period, self._publish_status_tick)
        self._set_status('WAITING_FOR_GOAL', event='manager_started')
        self.get_logger().info('Navigation manager ready')

    def _declare_parameters(self):
        self.declare_parameter('enabled', True)
        self.declare_parameter('status_frequency', 5.0)
        self.declare_parameter('goal_tolerance_xy', 0.25)
        self.declare_parameter('collision_distance', 0.15)
        self.declare_parameter('collision_risk_distance', 0.25)
        self.declare_parameter('no_valid_trajectory_timeout', 3.0)
        self.declare_parameter('publish_stop_on_collision', True)

    def _goal_callback(self, msg: PoseStamped):
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self.has_path = False
        self.planner_status = 'IDLE'
        self.controller_status = 'IDLE'
        self.last_no_valid_time = None
        self._set_status('PLANNING', event='goal_received')

    def _planner_status_callback(self, msg: String):
        self.planner_status = msg.data
        self._update_state()

    def _controller_status_callback(self, msg: String):
        self.controller_status = msg.data
        self._update_state()

    def _pose_callback(self, msg: PoseStamped):
        self.pose_xy = (msg.pose.position.x, msg.pose.position.y)
        self._update_state()

    def _path_callback(self, msg: Path):
        self.has_path = bool(msg.poses)
        self._update_state()

    def _scan_callback(self, msg: LaserScan):
        valid_ranges = [
            value for value in msg.ranges
            if math.isfinite(value) and msg.range_min <= value <= msg.range_max
        ]
        self.min_scan_range = min(valid_ranges) if valid_ranges else math.inf
        collision_status = 'CLEAR'
        if self.min_scan_range < self.collision_distance:
            collision_status = 'COLLISION'
        elif self.min_scan_range < self.collision_risk_distance:
            collision_status = 'COLLISION_RISK'
        self.collision_status_pub.publish(String(data=collision_status))
        self._update_state()

    def _update_state(self):
        if self.goal_xy is None:
            self._set_status('WAITING_FOR_GOAL')
            return

        if self.min_scan_range < self.collision_distance:
            self._set_status('COLLISION', event='collision_detected')
            if self.publish_stop_on_collision:
                self._publish_stop()
            return

        if self.min_scan_range < self.collision_risk_distance:
            self._set_status('COLLISION_RISK', event='collision_risk')
            return

        if self.planner_status.startswith('NO_PLAN'):
            self._set_status('NO_PLAN', event=self.planner_status)
            return

        if self.controller_status == 'NO_VALID_TRAJECTORY':
            now_sec = self._now_sec()
            if self.last_no_valid_time is None:
                self.last_no_valid_time = now_sec
            if now_sec - self.last_no_valid_time >= \
                    self.no_valid_trajectory_timeout:
                self._set_status(
                    'ABORTED',
                    event='no_valid_trajectory_timeout')
                self._publish_stop()
            else:
                self._set_status('NO_VALID_TRAJECTORY')
            return
        self.last_no_valid_time = None

        if self.controller_status == 'GOAL_REACHED' and self._pose_at_goal():
            self._set_status('SUCCESS', event='goal_reached')
            return

        if self.planner_status == 'SUCCESS' and self.has_path:
            self._set_status('FOLLOWING_PATH')
            return

        if self.planner_status == 'PLANNING':
            self._set_status('PLANNING')
            return

        self._set_status('PLANNING')

    def _pose_at_goal(self) -> bool:
        if self.goal_xy is None or self.pose_xy is None:
            return False
        return self._distance_xy(self.pose_xy, self.goal_xy) <= \
            self.goal_tolerance_xy

    def _set_status(self, status: str, event: Optional[str] = None):
        self.navigation_status = status
        if event is not None and event != self.last_event:
            self.event_pub.publish(String(data=event))
            self.last_event = event
            self.get_logger().info(f'Navigation event: {event}')
        if status != self.last_published_status:
            self.status_pub.publish(String(data=status))
            self.last_published_status = status
            self.get_logger().info(f'Navigation status: {status}')

    def _publish_status_tick(self):
        self.status_pub.publish(String(data=self.navigation_status))

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])


def main(args=None):
    rclpy.init(args=args)
    node = NavigationManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
