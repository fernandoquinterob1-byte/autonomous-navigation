import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker

from final_navigation.rrt_star import PlanningCostmap, RRTStarPlanner
from final_navigation.utils import yaw_to_quaternion


class RRTStarNode(Node):
    def __init__(self):
        super().__init__('rrt_star_node')
        self._declare_parameters()

        self.global_frame = self.get_parameter(
            'global_frame').get_parameter_value().string_value
        self.pose_topic = self.get_parameter(
            'pose_topic').get_parameter_value().string_value
        self.costmap_topic = self.get_parameter(
            'costmap_topic').get_parameter_value().string_value
        self.goal_topic = self.get_parameter(
            'goal_topic').get_parameter_value().string_value
        self.plan_on_start = self.get_parameter(
            'plan_on_start').get_parameter_value().bool_value
        self.default_goal_x = self.get_parameter(
            'default_goal_x').get_parameter_value().double_value
        self.default_goal_y = self.get_parameter(
            'default_goal_y').get_parameter_value().double_value
        self.publish_tree = self.get_parameter(
            'publish_tree').get_parameter_value().bool_value
        self.publish_nodes = self.get_parameter(
            'publish_nodes').get_parameter_value().bool_value

        self.planner = RRTStarPlanner(
            max_iterations=self.get_parameter(
                'max_iterations').get_parameter_value().integer_value,
            step_size=self.get_parameter(
                'step_size').get_parameter_value().double_value,
            goal_tolerance=self.get_parameter(
                'goal_tolerance').get_parameter_value().double_value,
            goal_sample_rate=self.get_parameter(
                'goal_sample_rate').get_parameter_value().double_value,
            search_radius=self.get_parameter(
                'search_radius').get_parameter_value().double_value,
            collision_check_resolution=self.get_parameter(
                'collision_check_resolution').get_parameter_value().double_value,
            occupied_threshold=self.get_parameter(
                'occupied_threshold').get_parameter_value().integer_value,
            planning_timeout=self.get_parameter(
                'planning_timeout').get_parameter_value().double_value,
        )

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.costmap_sub = self.create_subscription(
            OccupancyGrid, self.costmap_topic, self._costmap_callback,
            transient_qos)
        self.pose_sub = self.create_subscription(
            PoseStamped, self.pose_topic, self._pose_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, self.goal_topic, self._goal_callback, 10)

        self.path_pub = self.create_publisher(
            Path, '/final/global_path', transient_qos)
        self.tree_pub = self.create_publisher(
            Marker, '/final/rrt_tree', marker_qos)
        self.nodes_pub = self.create_publisher(
            Marker, '/final/rrt_nodes', marker_qos)
        self.status_pub = self.create_publisher(
            String, '/final/planner_status', transient_qos)

        self.costmap: Optional[PlanningCostmap] = None
        self.last_costmap_signature = None
        self.current_pose_xy: Optional[Tuple[float, float]] = None
        self.goal_xy: Optional[Tuple[float, float]] = None
        self.goal_yaw: Optional[float] = None
        self.goal_dirty = False
        self.is_planning = False

        self._publish_status('WAITING_FOR_COSTMAP')
        self.get_logger().info(
            f'RRT* waiting for {self.costmap_topic}, {self.pose_topic}, '
            f'and goals on {self.goal_topic}')

        if self.plan_on_start:
            self.goal_xy = (self.default_goal_x, self.default_goal_y)
            self.goal_yaw = 0.0
            self.goal_dirty = True
            self.get_logger().info(
                f'plan_on_start enabled: default goal '
                f'({self.default_goal_x:.2f}, {self.default_goal_y:.2f})')

    def _declare_parameters(self):
        self.declare_parameter('enabled', True)
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('pose_topic', '/ground_truth_pose')
        self.declare_parameter('costmap_topic', '/final/inflated_costmap')
        self.declare_parameter('goal_topic', '/final/goal_pose')
        self.declare_parameter('plan_on_start', False)
        self.declare_parameter('default_goal_x', 10.0)
        self.declare_parameter('default_goal_y', 0.0)
        self.declare_parameter('max_iterations', 3000)
        self.declare_parameter('step_size', 0.30)
        self.declare_parameter('goal_tolerance', 0.35)
        self.declare_parameter('goal_sample_rate', 0.10)
        self.declare_parameter('search_radius', 0.80)
        self.declare_parameter('collision_check_resolution', 0.05)
        self.declare_parameter('occupied_threshold', 50)
        self.declare_parameter('planning_timeout', 5.0)
        self.declare_parameter('publish_tree', True)
        self.declare_parameter('publish_nodes', True)

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
            occupied_threshold=self.planner.occupied_threshold,
        )
        signature = (
            self.costmap.width,
            self.costmap.height,
            self.costmap.resolution,
            msg.header.frame_id,
        )
        if signature != self.last_costmap_signature:
            self.get_logger().info(
                f'Received inflated costmap {self.costmap.width} x '
                f'{self.costmap.height}, resolution '
                f'{self.costmap.resolution:.3f}, frame={msg.header.frame_id}')
            self.last_costmap_signature = signature
        self._maybe_plan()

    def _pose_callback(self, msg: PoseStamped):
        if msg.header.frame_id and msg.header.frame_id not in (
                self.global_frame, 'world_demo'):
            self.get_logger().warn(
                f'Pose frame is {msg.header.frame_id}; assuming coordinates '
                f'are compatible with {self.global_frame}')
        self.current_pose_xy = (
            msg.pose.position.x,
            msg.pose.position.y,
        )
        self._maybe_plan()

    def _goal_callback(self, msg: PoseStamped):
        if msg.header.frame_id and msg.header.frame_id != self.global_frame:
            self.get_logger().warn(
                f'Goal frame is {msg.header.frame_id}; expected '
                f'{self.global_frame}. Assuming coordinates are compatible.')
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self.goal_yaw = self._yaw_from_quaternion(msg.pose.orientation)
        self.goal_dirty = True
        self.get_logger().info(
            f'Received goal ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}, '
            f'yaw={self.goal_yaw:.2f})')
        self._maybe_plan()

    def _maybe_plan(self):
        if self.is_planning:
            return
        if self.costmap is None:
            self._publish_status('WAITING_FOR_COSTMAP')
            return
        if self.current_pose_xy is None:
            self._publish_status('WAITING_FOR_POSE')
            return
        if self.goal_xy is None or not self.goal_dirty:
            self._publish_status('WAITING_FOR_GOAL')
            return

        self.is_planning = True
        self._publish_status('PLANNING')
        start = self.current_pose_xy
        goal = self.goal_xy
        self.get_logger().info(
            f'Planning RRT*: start=({start[0]:.2f}, {start[1]:.2f}), '
            f'goal=({goal[0]:.2f}, {goal[1]:.2f})')

        result = self.planner.plan(start, goal, self.costmap)
        self._publish_tree(result)
        self._publish_nodes(result)
        if result.status == 'SUCCESS':
            self._publish_path(result.path)
            self.get_logger().info(
                f'RRT* SUCCESS: {len(result.path)} path points, '
                f'{len(result.nodes)} nodes, {result.iterations} iterations, '
                f'{result.elapsed_time:.2f} s')
        else:
            self.get_logger().warn(
                f'RRT* failed with {result.status}: '
                f'{len(result.nodes)} nodes, {result.iterations} iterations, '
                f'{result.elapsed_time:.2f} s')
            self._publish_empty_path()

        self._publish_status(result.status)
        self.goal_dirty = False
        self.is_planning = False

    def _publish_path(self, points):
        msg = Path()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()

        for index, point in enumerate(points):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            yaw = 0.0
            if index == len(points) - 1 and self.goal_yaw is not None:
                yaw = self.goal_yaw
            elif index < len(points) - 1:
                next_point = points[index + 1]
                yaw = math.atan2(next_point[1] - point[1],
                                 next_point[0] - point[0])
            elif index > 0:
                previous = points[index - 1]
                yaw = math.atan2(point[1] - previous[1],
                                 point[0] - previous[0])
            pose.pose.orientation = yaw_to_quaternion(yaw)
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def _publish_empty_path(self):
        msg = Path()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(msg)

    def _publish_tree(self, result):
        if not self.publish_tree:
            return
        marker = self._base_marker('rrt_tree', 0, Marker.LINE_LIST)
        marker.scale.x = 0.025
        marker.color = ColorRGBA(r=0.1, g=0.55, b=1.0, a=0.55)
        for p1, p2 in result.edges:
            marker.points.append(Point(x=float(p1[0]), y=float(p1[1]), z=0.02))
            marker.points.append(Point(x=float(p2[0]), y=float(p2[1]), z=0.02))
        self.tree_pub.publish(marker)

    def _publish_nodes(self, result):
        if not self.publish_nodes:
            return
        marker = self._base_marker('rrt_nodes', 1, Marker.POINTS)
        marker.scale.x = 0.045
        marker.scale.y = 0.045
        marker.color = ColorRGBA(r=1.0, g=0.65, b=0.1, a=0.75)
        for node in result.nodes:
            marker.points.append(Point(x=float(node.x), y=float(node.y), z=0.04))
        self.nodes_pub.publish(marker)

    def _base_marker(self, namespace: str, marker_id: int, marker_type: int):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)

    @staticmethod
    def _yaw_from_quaternion(quat) -> float:
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None):
    rclpy.init(args=args)
    node = RRTStarNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
