import rclpy
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from final_navigation.costmap_2d import from_pgm_map
from final_navigation.inflation import inflate_costmap
from final_navigation.map_loader import load_map
from final_navigation.utils import count_values, yaw_to_quaternion


class CostmapNode(Node):
    def __init__(self):
        super().__init__('costmap_node')
        self._declare_parameters()

        self.global_frame = self.get_parameter(
            'global_frame').get_parameter_value().string_value
        self.map_yaml_path = self.get_parameter(
            'map_yaml_path').get_parameter_value().string_value
        self.publish_frequency = self.get_parameter(
            'publish_frequency').get_parameter_value().double_value

        self.occupied_cost = self.get_parameter(
            'occupied_cost').get_parameter_value().integer_value
        self.inflated_cost = self.get_parameter(
            'inflated_cost').get_parameter_value().integer_value
        self.free_cost = self.get_parameter(
            'free_cost').get_parameter_value().integer_value
        self.unknown_cost = self.get_parameter(
            'unknown_cost').get_parameter_value().integer_value
        self.robot_radius = self.get_parameter(
            'robot_radius').get_parameter_value().double_value
        self.inflation_radius = self.get_parameter(
            'inflation_radius').get_parameter_value().double_value

        pgm_map = load_map(self.map_yaml_path)
        self.costmap = from_pgm_map(
            pgm_map,
            occupied_cost=self.occupied_cost,
            free_cost=self.free_cost,
            unknown_cost=self.unknown_cost,
        )
        self.inflation_result = inflate_costmap(
            self.costmap,
            robot_radius=self.robot_radius,
            inflation_radius=self.inflation_radius,
            inflated_cost=self.inflated_cost,
            occupied_cost=self.occupied_cost,
            unknown_cost=self.unknown_cost,
        )

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid, '/final/occupancy_grid', map_qos)
        self.inflated_pub = self.create_publisher(
            OccupancyGrid, '/final/inflated_costmap', map_qos)

        self.occupancy_msg = self._make_grid_message(self.costmap.data)
        self.inflated_msg = self._make_grid_message(
            self.inflation_result.costmap.data)

        self._log_map_summary(pgm_map)
        timer_period = 1.0 / max(self.publish_frequency, 0.001)
        self.timer = self.create_timer(timer_period, self._publish_maps)
        self._publish_maps()

    def _declare_parameters(self):
        self.declare_parameter(
            'map_yaml_path',
            'package://amcl_launch/map/depot.yaml')
        self.declare_parameter('global_frame', 'map')
        self.declare_parameter('robot_base_frame', 'base_link')
        self.declare_parameter('robot_radius', 0.30)
        self.declare_parameter('inflation_radius', 0.40)
        self.declare_parameter('occupied_cost', 100)
        self.declare_parameter('inflated_cost', 80)
        self.declare_parameter('free_cost', 0)
        self.declare_parameter('unknown_cost', -1)
        self.declare_parameter('publish_frequency', 1.0)

    def _make_grid_message(self, data):
        msg = OccupancyGrid()
        msg.header.frame_id = self.global_frame
        msg.info = MapMetaData()
        msg.info.resolution = self.costmap.resolution
        msg.info.width = self.costmap.width
        msg.info.height = self.costmap.height
        msg.info.origin.position.x = float(self.costmap.origin[0])
        msg.info.origin.position.y = float(self.costmap.origin[1])
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = yaw_to_quaternion(self.costmap.origin[2])
        msg.data = [int(value) for value in data]
        return msg

    def _publish_maps(self):
        stamp = self.get_clock().now().to_msg()
        self.occupancy_msg.header.stamp = stamp
        self.inflated_msg.header.stamp = stamp
        self.occupancy_pub.publish(self.occupancy_msg)
        self.inflated_pub.publish(self.inflated_msg)

    def _log_map_summary(self, pgm_map):
        free, occupied, unknown, inflated = count_values(self.costmap.data)
        inf_free, inf_occupied, inf_unknown, inf_inflated = count_values(
            self.inflation_result.costmap.data)
        width_m = self.costmap.width * self.costmap.resolution
        height_m = self.costmap.height * self.costmap.resolution

        self.get_logger().info(f'Map loaded: {pgm_map.metadata.yaml_path}')
        self.get_logger().info(f'Image loaded: {pgm_map.metadata.image_path}')
        self.get_logger().info(
            f'Size: {self.costmap.width} x {self.costmap.height} cells '
            f'({width_m:.2f} m x {height_m:.2f} m)')
        self.get_logger().info(
            f'Resolution: {self.costmap.resolution:.3f} m/cell')
        self.get_logger().info(
            f'Origin: x={self.costmap.origin[0]:.3f}, '
            f'y={self.costmap.origin[1]:.3f}, '
            f'yaw={self.costmap.origin[2]:.3f}')
        self.get_logger().info(
            f'Raw occupancy counts: free={free}, occupied={occupied}, '
            f'unknown={unknown}, inflated={inflated}')
        self.get_logger().info(
            f'Inflation: robot_radius={self.robot_radius:.2f} m, '
            f'inflation_radius={self.inflation_radius:.2f} m, '
            f'effective_radius={self.inflation_result.effective_radius:.2f} m, '
            f'cells={self.inflation_result.inflation_radius_cells}')
        self.get_logger().info(
            f'Inflated map counts: free={inf_free}, '
            f'occupied={inf_occupied}, unknown={inf_unknown}, '
            f'inflated={inf_inflated}')
        self.get_logger().info(
            'Publishing /final/occupancy_grid and /final/inflated_costmap')


def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
