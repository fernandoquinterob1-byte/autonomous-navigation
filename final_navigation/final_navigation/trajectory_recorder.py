import csv
import json
import math
import os
from datetime import datetime
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker


class TrajectoryRecorder(Node):
    def __init__(self):
        super().__init__('trajectory_recorder')
        self._declare_parameters()

        self.global_frame = self.get_parameter(
            'global_frame').get_parameter_value().string_value
        self.publish_frequency = self.get_parameter(
            'publish_frequency').get_parameter_value().double_value
        self.save_csv = self.get_parameter(
            'save_csv').get_parameter_value().bool_value
        self.output_dir = self.get_parameter(
            'output_dir').get_parameter_value().string_value
        self.sample_min_distance = self.get_parameter(
            'sample_min_distance').get_parameter_value().double_value
        self.save_plots = self.get_parameter(
            'save_plots').get_parameter_value().bool_value
        self.figures_dir = self.get_parameter(
            'figures_dir').get_parameter_value().string_value

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped, '/ground_truth_pose', self._pose_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/final/goal_pose', self._goal_callback, 10)
        self.controller_status_sub = self.create_subscription(
            String, '/final/controller_status',
            self._controller_status_callback, 10)
        self.planner_status_sub = self.create_subscription(
            String, '/final/planner_status', self._planner_status_callback, 10)
        self.navigation_status_sub = self.create_subscription(
            String, '/final/navigation_status',
            self._navigation_status_callback, 10)
        self.navigation_event_sub = self.create_subscription(
            String, '/final/navigation_event',
            self._navigation_event_callback, 10)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/final/inflated_costmap',
            self._costmap_callback, transient_qos)
        self.global_path_sub = self.create_subscription(
            Path, '/final/global_path',
            self._global_path_callback, transient_qos)
        self.selected_trajectory_sub = self.create_subscription(
            Marker, '/final/selected_trajectory',
            self._selected_trajectory_callback, transient_qos)
        self.rrt_tree_sub = self.create_subscription(
            Marker, '/final/rrt_tree',
            self._rrt_tree_callback, transient_qos)
        self.rrt_nodes_sub = self.create_subscription(
            Marker, '/final/rrt_nodes',
            self._rrt_nodes_callback, transient_qos)

        self.trajectory_pub = self.create_publisher(
            Path, '/final/robot_trajectory', transient_qos)
        self.metrics_pub = self.create_publisher(
            String, '/final/navigation_metrics', transient_qos)

        self.goal_xy: Optional[Tuple[float, float]] = None
        self.current_pose: Optional[PoseStamped] = None
        self.previous_xy: Optional[Tuple[float, float]] = None
        self.trajectory = Path()
        self.trajectory.header.frame_id = self.global_frame
        self.trajectory_length = 0.0
        self.min_distance_to_goal = math.inf
        self.navigation_start_time: Optional[float] = None
        self.planner_status = 'IDLE'
        self.controller_status = 'IDLE'
        self.navigation_status = 'IDLE'
        self.navigation_event = ''
        self.final_status = 'IDLE'
        self.run_id = ''
        self.csv_path = ''
        self.csv_file = None
        self.csv_writer = None
        self.samples_saved = 0
        self.last_saved_xy: Optional[Tuple[float, float]] = None
        self.last_logged_final_status: Optional[str] = None
        self.last_costmap: Optional[OccupancyGrid] = None
        self.global_path = Path()
        self.selected_trajectory_points = []
        self.rrt_tree_segments = []
        self.rrt_nodes = []
        self.plots_saved = False

        if self.save_csv:
            os.makedirs(self.output_dir, exist_ok=True)
        if self.save_plots:
            os.makedirs(self.figures_dir, exist_ok=True)

        period = 1.0 / max(self.publish_frequency, 0.001)
        self.timer = self.create_timer(period, self._publish_outputs)
        self.get_logger().info('Trajectory recorder ready')

    def _declare_parameters(self):
        self.declare_parameter('enabled', True)
        self.declare_parameter('publish_frequency', 5.0)
        self.declare_parameter('save_csv', True)
        self.declare_parameter('sample_min_distance', 0.01)
        self.declare_parameter('save_plots', True)
        self.declare_parameter(
            'output_dir',
            '/home/fernando-quintero/Descargas/navegacion/'
            'final_navigation/results')
        self.declare_parameter(
            'figures_dir',
            '/home/fernando-quintero/Descargas/navegacion/'
            'final_navigation/results/figures')
        self.declare_parameter('global_frame', 'map')

    def _goal_callback(self, msg: PoseStamped):
        self.goal_xy = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f'New goal received: ({self.goal_xy[0]:.2f}, '
            f'{self.goal_xy[1]:.2f})')
        self._reset_run()
        self._record_event_sample('goal_received')

    def _pose_callback(self, msg: PoseStamped):
        self.current_pose = msg
        if self.goal_xy is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.navigation_start_time is None:
            self.navigation_start_time = now_sec

        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = self._yaw_from_quaternion(msg.pose.orientation)
        xy = (x, y)

        distance_to_goal = self._distance_to_goal(xy)
        self.min_distance_to_goal = min(self.min_distance_to_goal,
                                        distance_to_goal)

        if not self._should_save_pose_sample(xy):
            return

        if self.previous_xy is not None:
            self.trajectory_length += self._distance_xy(self.previous_xy, xy)
        self.previous_xy = xy
        self.last_saved_xy = xy

        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = msg.header.stamp
        pose.pose = msg.pose
        self.trajectory.header.stamp = msg.header.stamp
        self.trajectory.poses.append(pose)

        self._write_csv_row(
            now_sec, x, y, yaw, distance_to_goal, event='pose_sample')

    def _planner_status_callback(self, msg: String):
        if msg.data == self.planner_status:
            return
        self.planner_status = msg.data
        if msg.data.startswith('NO_PLAN'):
            self.final_status = msg.data
            self._record_event_sample(f'planner_{msg.data}')
            self._log_final_summary()

    def _controller_status_callback(self, msg: String):
        if msg.data == self.controller_status:
            return
        self.controller_status = msg.data
        if msg.data in ('GOAL_REACHED', 'STOPPED', 'NO_VALID_TRAJECTORY'):
            self.final_status = msg.data
            self._record_event_sample(f'controller_{msg.data}')
            if msg.data != 'NO_VALID_TRAJECTORY':
                self._log_final_summary()

    def _navigation_status_callback(self, msg: String):
        if msg.data == self.navigation_status:
            return
        self.navigation_status = msg.data
        if msg.data in ('SUCCESS', 'NO_PLAN', 'COLLISION', 'ABORTED'):
            if not self.final_status.startswith('NO_PLAN'):
                self.final_status = msg.data
            self._record_event_sample(f'navigation_{msg.data}')
            self._log_final_summary()

    def _navigation_event_callback(self, msg: String):
        if msg.data == self.navigation_event:
            return
        self.navigation_event = msg.data
        self._record_event_sample(f'event_{msg.data}')

    def _costmap_callback(self, msg: OccupancyGrid):
        self.last_costmap = msg

    def _global_path_callback(self, msg: Path):
        self.global_path = msg

    def _selected_trajectory_callback(self, msg: Marker):
        self.selected_trajectory_points = [
            (point.x, point.y)
            for point in msg.points
        ]

    def _rrt_tree_callback(self, msg: Marker):
        points = [(point.x, point.y) for point in msg.points]
        self.rrt_tree_segments = [
            (points[index], points[index + 1])
            for index in range(0, len(points) - 1, 2)
        ]

    def _rrt_nodes_callback(self, msg: Marker):
        self.rrt_nodes = [
            (point.x, point.y)
            for point in msg.points
        ]

    def _reset_run(self):
        self._close_csv()
        self.trajectory = Path()
        self.trajectory.header.frame_id = self.global_frame
        self.trajectory_length = 0.0
        self.min_distance_to_goal = math.inf
        self.navigation_start_time = self.get_clock().now().nanoseconds * 1e-9
        self.previous_xy = None
        self.last_saved_xy = None
        self.final_status = 'ACTIVE'
        self.samples_saved = 0
        self.last_logged_final_status = None
        self.global_path = Path()
        self.selected_trajectory_points = []
        self.rrt_tree_segments = []
        self.rrt_nodes = []
        self.plots_saved = False
        if self.save_csv:
            self.run_id, self.csv_path = self._new_csv_path()
            self.csv_file = open(
                self.csv_path, 'x', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'time',
                'x',
                'y',
                'yaw',
                'goal_x',
                'goal_y',
                'distance_to_goal',
                'planner_status',
                'controller_status',
                'navigation_status',
                'event',
                'trajectory_length',
                'run_id',
            ])
            self.get_logger().info(
                f'Created trajectory CSV: {self.csv_path}')
        self.get_logger().info('Trajectory reset for new goal')

    def _publish_outputs(self):
        self.trajectory.header.stamp = self.get_clock().now().to_msg()
        self.trajectory_pub.publish(self.trajectory)
        metrics = self._metrics()
        self.metrics_pub.publish(String(data=json.dumps(metrics)))

    def _metrics(self):
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        navigation_time = 0.0
        if self.navigation_start_time is not None:
            navigation_time = max(0.0, now_sec - self.navigation_start_time)
        final_error = math.inf
        if self.current_pose is not None:
            final_error = self._distance_to_goal((
                self.current_pose.pose.position.x,
                self.current_pose.pose.position.y,
            ))
        return {
            'status': self.final_status,
            'navigation_status': self.navigation_status,
            'planner_status': self.planner_status,
            'controller_status': self.controller_status,
            'navigation_time': round(navigation_time, 3),
            'trajectory_length': round(self.trajectory_length, 3),
            'final_error': None if not math.isfinite(final_error)
            else round(final_error, 3),
            'min_distance_to_goal': None if not math.isfinite(
                self.min_distance_to_goal) else round(
                    self.min_distance_to_goal, 3),
            'samples': len(self.trajectory.poses),
            'goal_x': None if self.goal_xy is None else round(
                self.goal_xy[0], 3),
            'goal_y': None if self.goal_xy is None else round(
                self.goal_xy[1], 3),
            'run_id': self.run_id,
            'csv_path': self.csv_path,
        }

    def _write_csv_row(
        self,
        now_sec: float,
        x: float,
        y: float,
        yaw: float,
        distance_to_goal: float,
        event: str,
    ):
        if self.csv_writer is None:
            return
        goal_x = self.goal_xy[0] if self.goal_xy is not None else ''
        goal_y = self.goal_xy[1] if self.goal_xy is not None else ''
        self.samples_saved += 1
        self.csv_writer.writerow([
            f'{now_sec:.6f}',
            f'{x:.6f}',
            f'{y:.6f}',
            f'{yaw:.6f}',
            goal_x,
            goal_y,
            '' if not math.isfinite(distance_to_goal)
            else f'{distance_to_goal:.6f}',
            self.planner_status,
            self.controller_status,
            self.navigation_status,
            event,
            f'{self.trajectory_length:.6f}',
            self.run_id,
        ])
        self.csv_file.flush()
        if self.samples_saved == 1 or self.samples_saved % 25 == 0:
            self.get_logger().info(
                f'Saved CSV sample {self.samples_saved} '
                f'for run {self.run_id}')

    def _record_event_sample(self, event: str):
        if self.goal_xy is None:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.current_pose is None:
            self._write_blank_csv_row(now_sec, event)
            return
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        yaw = self._yaw_from_quaternion(self.current_pose.pose.orientation)
        self._write_csv_row(
            now_sec, x, y, yaw, self._distance_to_goal((x, y)), event)

    def _write_blank_csv_row(self, now_sec: float, event: str):
        if self.csv_writer is None:
            return
        goal_x = self.goal_xy[0] if self.goal_xy is not None else ''
        goal_y = self.goal_xy[1] if self.goal_xy is not None else ''
        self.samples_saved += 1
        self.csv_writer.writerow([
            f'{now_sec:.6f}',
            '',
            '',
            '',
            goal_x,
            goal_y,
            '',
            self.planner_status,
            self.controller_status,
            self.navigation_status,
            event,
            f'{self.trajectory_length:.6f}',
            self.run_id,
        ])
        self.csv_file.flush()
        self.get_logger().info(
            f'Saved CSV event sample {self.samples_saved}: {event}')

    def _should_save_pose_sample(self, xy: Tuple[float, float]) -> bool:
        if self.last_saved_xy is None:
            return True
        return self._distance_xy(self.last_saved_xy, xy) >= \
            self.sample_min_distance

    def _new_csv_path(self) -> Tuple[str, str]:
        goal_x = 0.0 if self.goal_xy is None else self.goal_xy[0]
        goal_y = 0.0 if self.goal_xy is None else self.goal_xy[1]
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        goal_label = self._format_goal_label(goal_x, goal_y)
        for suffix in range(1000):
            suffix_label = '' if suffix == 0 else f'_{suffix:03d}'
            run_id = f'{stamp}_goal_{goal_label}{suffix_label}'
            path = os.path.join(
                self.output_dir, f'navigation_{run_id}.csv')
            if not os.path.exists(path):
                return run_id, path
        raise RuntimeError('Unable to allocate unique trajectory CSV path')

    @staticmethod
    def _format_goal_label(goal_x: float, goal_y: float) -> str:
        def clean(value: float) -> str:
            return f'{value:+.2f}'.replace('+', 'p').replace('-', 'm').replace(
                '.', 'p')
        return f'x{clean(goal_x)}_y{clean(goal_y)}'

    def _log_final_summary(self):
        if self.final_status == self.last_logged_final_status:
            return
        if not self._is_terminal_status(self.final_status):
            return
        self.last_logged_final_status = self.final_status
        metrics = self._metrics()
        self.get_logger().info(
            f'Final status detected: {self.final_status}')
        self.get_logger().info(
            f'Navigation metrics summary: {json.dumps(metrics)}')
        self._save_result_plots()

    def _save_result_plots(self):
        if not self.save_plots or self.plots_saved or not self.run_id:
            return
        self.plots_saved = True
        if self.last_costmap is None:
            self.get_logger().warn(
                'Skipping result plots: no inflated costmap received')
            return
        try:
            self._save_global_planning_plot()
            self._save_local_planning_plot()
        except Exception as exc:  # pragma: no cover - diagnostic path
            self.get_logger().warn(f'Failed to save result plots: {exc}')

    def _save_global_planning_plot(self):
        plot_data = self._prepare_costmap_plot_data()
        if plot_data is None:
            return
        plt = plot_data['plt']
        fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
        image = ax.imshow(
            plot_data['grid'],
            cmap=plot_data['cmap'],
            norm=plot_data['norm'],
            origin='lower',
            extent=plot_data['extent'],
            interpolation='nearest',
        )
        self._style_costmap_axis(ax, 'Planificacion global sobre costmap')
        self._plot_rrt_tree(ax)
        self._plot_rrt_nodes(ax)
        self._plot_path(ax, self.global_path.poses, 'tab:blue',
                        'camino seleccionado', linewidth=3.0)
        self._plot_path(ax, self.trajectory.poses, 'tab:green',
                        'trayectoria real', linewidth=2.5)
        self._plot_goal_and_final_pose(ax)
        self._set_full_plot_bounds(ax, plot_data['extent'])
        cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3],
                            fraction=0.03, pad=0.02)
        cbar.ax.set_yticklabels(
            ['libre', 'desconocido', 'inflado', 'ocupado'])
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        path = os.path.join(
            self.figures_dir, f'global_planning_{self.run_id}.png')
        fig.savefig(path)
        plt.close(fig)
        self.get_logger().info(f'Saved global planning plot: {path}')
        self.get_logger().info(
            f'Global plot RRT*: {len(self.rrt_tree_segments)} edges, '
            f'{len(self.rrt_nodes)} nodes')

    def _save_local_planning_plot(self):
        if not self.selected_trajectory_points:
            self.get_logger().info(
                'Skipping local planning plot: no selected trajectory received')
            return
        plot_data = self._prepare_costmap_plot_data()
        if plot_data is None:
            return
        plt = plot_data['plt']
        fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
        ax.imshow(
            plot_data['grid'],
            cmap=plot_data['cmap'],
            norm=plot_data['norm'],
            origin='lower',
            extent=plot_data['extent'],
            interpolation='nearest',
        )
        self._style_costmap_axis(ax, 'Planificacion local DWA')
        if self.global_path.poses:
            self._plot_path(ax, self.global_path.poses, 'tab:blue',
                            'path global', linewidth=1.5, alpha=0.7)
        if self.trajectory.poses:
            self._plot_path(ax, self.trajectory.poses, 'tab:green',
                            'trayectoria real', linewidth=1.8)
        xs = [point[0] for point in self.selected_trajectory_points]
        ys = [point[1] for point in self.selected_trajectory_points]
        ax.plot(xs, ys, color='tab:red', linewidth=2.2,
                label='trayectoria local seleccionada')
        self._plot_goal_and_final_pose(ax)
        self._set_plot_bounds(
            ax, plot_data['extent'], extra_points=self.selected_trajectory_points)
        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        path = os.path.join(
            self.figures_dir, f'local_planning_{self.run_id}.png')
        fig.savefig(path)
        plt.close(fig)
        self.get_logger().info(f'Saved local planning plot: {path}')

    def _prepare_costmap_plot_data(self):
        try:
            os.environ.setdefault(
                'MPLCONFIGDIR', '/tmp/matplotlib-final-navigation')
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import BoundaryNorm, ListedColormap
        except ImportError as exc:
            self.get_logger().warn(
                f'Skipping result plots: matplotlib/numpy unavailable: {exc}')
            return None

        msg = self.last_costmap
        width = msg.info.width
        height = msg.info.height
        values = np.array(msg.data, dtype=np.int16).reshape((height, width))
        grid = np.zeros_like(values, dtype=np.uint8)
        grid[values == -1] = 1
        grid[values >= 80] = 2
        grid[values >= 100] = 3
        cmap = ListedColormap(['white', '0.65', 'orange', 'black'])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        extent = [
            origin_x,
            origin_x + width * msg.info.resolution,
            origin_y,
            origin_y + height * msg.info.resolution,
        ]
        return {
            'plt': plt,
            'grid': grid,
            'cmap': cmap,
            'norm': norm,
            'extent': extent,
        }

    @staticmethod
    def _style_costmap_axis(ax, title: str):
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color='tab:blue', alpha=0.18, linewidth=0.4)

    @staticmethod
    def _plot_path(ax, poses, color: str, label: str, linewidth: float,
                   alpha: float = 1.0):
        if not poses:
            return
        xs = [pose.pose.position.x for pose in poses]
        ys = [pose.pose.position.y for pose in poses]
        ax.plot(xs, ys, color=color, linewidth=linewidth,
                alpha=alpha, label=label)

    def _plot_rrt_tree(self, ax):
        if not self.rrt_tree_segments:
            return
        for index, (start, end) in enumerate(self.rrt_tree_segments):
            label = 'arbol RRT*' if index == 0 else None
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color='black',
                linewidth=0.85,
                alpha=0.75,
                label=label,
                zorder=2,
            )

    def _plot_rrt_nodes(self, ax):
        if not self.rrt_nodes:
            return
        xs = [point[0] for point in self.rrt_nodes]
        ys = [point[1] for point in self.rrt_nodes]
        ax.scatter(
            xs,
            ys,
            c='tab:red',
            s=9,
            alpha=0.75,
            linewidths=0,
            label='nodos RRT*',
            zorder=3,
        )

    def _plot_goal_and_final_pose(self, ax):
        if self.goal_xy is not None:
            ax.scatter([self.goal_xy[0]], [self.goal_xy[1]], c='tab:purple',
                       s=55, marker='x', label='goal solicitado')
        if self.trajectory.poses:
            start = self.trajectory.poses[0].pose.position
            end = self.trajectory.poses[-1].pose.position
            ax.scatter([start.x], [start.y], c='tab:green', s=40,
                       marker='o', label='inicio')
            ax.scatter([end.x], [end.y], c='tab:red', s=45,
                       marker='o', label='pose final')

    def _set_plot_bounds(self, ax, extent, extra_points=None):
        points = []
        points.extend((pose.pose.position.x, pose.pose.position.y)
                      for pose in self.global_path.poses)
        points.extend((pose.pose.position.x, pose.pose.position.y)
                      for pose in self.trajectory.poses)
        if self.goal_xy is not None:
            points.append(self.goal_xy)
        if extra_points:
            points.extend(extra_points)
        if not points:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            return
        margin = 1.0
        x_min = max(extent[0], min(point[0] for point in points) - margin)
        x_max = min(extent[1], max(point[0] for point in points) + margin)
        y_min = max(extent[2], min(point[1] for point in points) - margin)
        y_max = min(extent[3], max(point[1] for point in points) + margin)
        if x_max <= x_min or y_max <= y_min:
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            return
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    @staticmethod
    def _set_full_plot_bounds(ax, extent):
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    def _distance_to_goal(self, xy: Tuple[float, float]) -> float:
        if self.goal_xy is None:
            return math.inf
        return self._distance_xy(xy, self.goal_xy)

    def _close_csv(self):
        if self.csv_file is not None:
            self.get_logger().info(f'Closing trajectory CSV: {self.csv_path}')
            self.csv_file.close()
        self.csv_file = None
        self.csv_writer = None

    @staticmethod
    def _is_terminal_status(status: str) -> bool:
        return status in (
            'GOAL_REACHED',
            'SUCCESS',
            'NO_PLAN',
            'COLLISION',
            'ABORTED',
            'STOPPED',
        ) or status.startswith('NO_PLAN')

    @staticmethod
    def _yaw_from_quaternion(quat) -> float:
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _distance_xy(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def destroy_node(self):
        self._close_csv()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
