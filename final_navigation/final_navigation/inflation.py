import math
from dataclasses import dataclass
from typing import List

from final_navigation.costmap_2d import Costmap2D


@dataclass(frozen=True)
class InflationResult:
    costmap: Costmap2D
    inflation_radius_cells: int
    effective_radius: float


def inflate_costmap(
    costmap: Costmap2D,
    robot_radius: float = 0.30,
    inflation_radius: float = 0.40,
    inflated_cost: int = 80,
    occupied_cost: int = 100,
    unknown_cost: int = -1,
) -> InflationResult:
    effective_radius = max(robot_radius, inflation_radius)
    radius_cells = int(math.ceil(effective_radius / costmap.resolution))
    inflated: List[int] = list(costmap.data)

    occupied_indices = [
        index for index, value in enumerate(costmap.data)
        if value >= occupied_cost
    ]

    for index in occupied_indices:
        obstacle_row = index // costmap.width
        obstacle_col = index % costmap.width

        for drow in range(-radius_cells, radius_cells + 1):
            for dcol in range(-radius_cells, radius_cells + 1):
                distance = math.hypot(dcol, drow) * costmap.resolution
                if distance > effective_radius:
                    continue

                row = obstacle_row + drow
                col = obstacle_col + dcol
                if not costmap.is_inside(col, row):
                    continue

                target_index = costmap.index(col, row)
                current_value = inflated[target_index]
                if current_value == unknown_cost or current_value >= occupied_cost:
                    continue
                inflated[target_index] = max(current_value, inflated_cost)

    inflated_map = Costmap2D(
        width=costmap.width,
        height=costmap.height,
        resolution=costmap.resolution,
        origin=costmap.origin,
        data=inflated,
        free_cost=costmap.free_cost,
        occupied_cost=costmap.occupied_cost,
        unknown_cost=costmap.unknown_cost,
    )
    return InflationResult(inflated_map, radius_cells, effective_radius)
