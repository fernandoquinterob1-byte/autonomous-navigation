import math
from typing import Iterable, Tuple

from geometry_msgs.msg import Quaternion


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def yaw_to_quaternion(yaw: float) -> Quaternion:
    quat = Quaternion()
    quat.z = math.sin(yaw * 0.5)
    quat.w = math.cos(yaw * 0.5)
    return quat


def count_values(values: Iterable[int]) -> Tuple[int, int, int, int]:
    free = occupied = unknown = inflated = 0
    for value in values:
        if value == 0:
            free += 1
        elif value == 100:
            occupied += 1
        elif value < 0:
            unknown += 1
        else:
            inflated += 1
    return free, occupied, unknown, inflated
