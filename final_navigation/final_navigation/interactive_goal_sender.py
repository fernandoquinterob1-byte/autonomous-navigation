import json
import math
import threading
import time
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


NAVIGATION_TERMINAL_STATES = {
    'SUCCESS',
    'GOAL_REACHED',
    'NO_PLAN',
    'COLLISION',
    'ABORTED',
}


class InteractiveGoalSender(Node):
    def __init__(self):
        super().__init__('interactive_goal_sender')
        self.declare_parameter('goal_timeout', 120.0)
        self.declare_parameter('global_frame', 'map')

        self.goal_timeout = self.get_parameter(
            'goal_timeout').get_parameter_value().double_value
        self.global_frame = self.get_parameter(
            'global_frame').get_parameter_value().string_value

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.goal_pub = self.create_publisher(
            PoseStamped, '/final/goal_pose', 10)
        self.stop_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(
            String, '/final/planner_status',
            lambda msg: self._status_callback('PLANNER', msg.data),
            transient_qos)
        self.create_subscription(
            String, '/final/controller_status',
            lambda msg: self._status_callback('CONTROLLER', msg.data),
            transient_qos)
        self.create_subscription(
            String, '/final/navigation_status',
            lambda msg: self._status_callback('NAVIGATION', msg.data),
            transient_qos)
        self.create_subscription(
            String, '/final/navigation_event',
            lambda msg: self._status_callback('EVENT', msg.data),
            transient_qos)
        self.create_subscription(
            String, '/final/navigation_metrics',
            self._metrics_callback,
            transient_qos)
        self.create_subscription(
            PoseStamped, '/ground_truth_pose',
            self._pose_callback,
            10)

        self.lock = threading.Lock()
        self.active_goal: Optional[Tuple[float, float, float]] = None
        self.goal_sent_wall_time: Optional[float] = None
        self.last_printed = {}
        self.last_metrics = ''
        self.current_pose: Optional[Tuple[float, float, float]] = None
        self.terminal_seen = False
        self.terminal_summary_printed = False
        self.metrics_printed_for_goal = False

        self.create_timer(0.5, self._timeout_tick)

    def send_goal(self, x: float, y: float, yaw: float):
        self._wait_for_goal_subscribers()

        msg = PoseStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        with self.lock:
            self.active_goal = (x, y, yaw)
            self.goal_sent_wall_time = time.monotonic()
            self.last_printed = {}
            self.last_metrics = ''
            self.terminal_seen = False
            self.terminal_summary_printed = False
            self.metrics_printed_for_goal = False

        self.goal_pub.publish(msg)
        print(f'[GOAL SENT] x={x:.3f} y={y:.3f} yaw={yaw:.3f}', flush=True)

    def publish_stop(self):
        self.stop_pub.publish(Twist())
        print('[STOP] Published zero /cmd_vel', flush=True)

    def _wait_for_goal_subscribers(self):
        deadline = time.monotonic() + 2.0
        while self.goal_pub.get_subscription_count() == 0 and \
                time.monotonic() < deadline and rclpy.ok():
            time.sleep(0.05)
        if self.goal_pub.get_subscription_count() == 0:
            print(
                '[WARN] /final/goal_pose has no subscribers yet',
                flush=True)

    def _status_callback(self, label: str, status: str):
        should_print = False
        terminal = False
        should_print_summary = False
        with self.lock:
            if self.last_printed.get(label) != status:
                self.last_printed[label] = status
                should_print = True
            terminal = self._is_terminal_status(label, status)
            if terminal:
                self.terminal_seen = True
                if not self.terminal_summary_printed:
                    self.terminal_summary_printed = True
                    should_print_summary = True

        if should_print:
            print(f'[{label}] {status}', flush=True)
        if should_print_summary:
            self._print_final_result(status)

    def _metrics_callback(self, msg: String):
        should_print = False
        with self.lock:
            self.last_metrics = msg.data
            if self.terminal_seen and not self.metrics_printed_for_goal:
                self.metrics_printed_for_goal = True
                should_print = True
        if should_print:
            print(f'[METRICS] {self._compact_metrics(msg.data)}', flush=True)

    def _pose_callback(self, msg: PoseStamped):
        yaw = self._yaw_from_quaternion(msg.pose.orientation)
        with self.lock:
            self.current_pose = (
                msg.pose.position.x,
                msg.pose.position.y,
                yaw,
            )

    def _timeout_tick(self):
        with self.lock:
            if self.active_goal is None or self.goal_sent_wall_time is None:
                return
            if self.terminal_seen:
                self.active_goal = None
                self.goal_sent_wall_time = None
                return
            elapsed = time.monotonic() - self.goal_sent_wall_time
            timed_out = elapsed > self.goal_timeout
        if timed_out:
            self.publish_stop()
            print(f'[TIMEOUT] Goal exceeded {self.goal_timeout:.1f} s',
                  flush=True)
            with self.lock:
                self.active_goal = None
                self.goal_sent_wall_time = None

    def _print_final_result(self, terminal_status: str):
        with self.lock:
            pose = self.current_pose
            goal = self.active_goal

        print('[FINAL RESULT]', flush=True)
        print(f'  terminal_status: {terminal_status}', flush=True)

        if goal is None:
            print('  requested_goal: unavailable', flush=True)
        else:
            print(
                f'  requested_goal: x={goal[0]:.3f} y={goal[1]:.3f} '
                f'yaw={goal[2]:.3f}',
                flush=True)

        if pose is None:
            print('  ground_truth_pose: unavailable', flush=True)
            print('  euclidean_error: unavailable', flush=True)
            return

        print(
            f'  ground_truth_pose: x={pose[0]:.3f} y={pose[1]:.3f} '
            f'yaw={pose[2]:.3f}',
            flush=True)
        if goal is None:
            print('  euclidean_error: unavailable', flush=True)
            return

        distance = math.hypot(goal[0] - pose[0], goal[1] - pose[1])
        yaw_error = self._normalize_angle(goal[2] - pose[2])
        print(f'  euclidean_error: {distance:.3f} m', flush=True)
        print(f'  yaw_error: {yaw_error:.3f} rad', flush=True)

    @staticmethod
    def _is_terminal_status(label: str, status: str) -> bool:
        if label == 'NAVIGATION':
            return status in NAVIGATION_TERMINAL_STATES
        if label == 'PLANNER':
            return status.startswith('NO_PLAN')
        return False

    @staticmethod
    def _compact_metrics(data: str) -> str:
        try:
            metrics = json.loads(data)
        except json.JSONDecodeError:
            return data
        keys = [
            'status',
            'navigation_status',
            'planner_status',
            'controller_status',
            'navigation_time',
            'trajectory_length',
            'final_error',
            'samples',
            'csv_path',
        ]
        compact = {
            key: metrics[key]
            for key in keys
            if key in metrics
        }
        return json.dumps(compact)

    @staticmethod
    def _yaw_from_quaternion(quat) -> float:
        siny_cosp = 2.0 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def _print_help():
    print()
    print('Interactive Goal Sender')
    print('Escribe: x y yaw')
    print('Comandos:')
    print('  simple   -> goal (8.0, 0.0, 0.0)')
    print('  hard     -> goal (14.0, 2.0, 0.0)')
    print('  invalid  -> goal (50.0, 50.0, 0.0)')
    print('  stop     -> publica cmd_vel cero')
    print('  h/help   -> mostrar ayuda')
    print('  q        -> salir')
    print()


def _parse_goal(text: str) -> Optional[Tuple[float, float, float]]:
    aliases = {
        'simple': (8.0, 0.0, 0.0),
        'hard': (14.0, 2.0, 0.0),
        'invalid': (50.0, 50.0, 0.0),
    }
    command = text.strip().lower()
    if command in aliases:
        return aliases[command]

    parts = text.split()
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _input_loop(node: InteractiveGoalSender):
    _print_help()
    while rclpy.ok():
        try:
            text = input('Ingrese goal x y yaw:\n> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = text.lower()
        if command in ('q', 'quit', 'exit'):
            break
        if command in ('h', 'help', '?'):
            _print_help()
            continue
        if command == 'stop':
            node.publish_stop()
            continue

        goal = _parse_goal(text)
        if goal is None:
            print('[INPUT ERROR] Usa: x y yaw, o simple/hard/invalid/stop/q',
                  flush=True)
            continue

        node.send_goal(*goal)


def main(args=None):
    rclpy.init(args=args)
    node = InteractiveGoalSender()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        _input_loop(node)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
