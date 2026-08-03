from dataclasses import dataclass
from typing import List

from final_navigation.map_loader import PgmMap


@dataclass(frozen=True)
class Costmap2D:
    width: int
    height: int
    resolution: float
    origin: tuple
    data: List[int]
    free_cost: int = 0
    occupied_cost: int = 100
    unknown_cost: int = -1

    def index(self, col: int, row: int) -> int:
        return row * self.width + col

    def is_inside(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def cost_at(self, col: int, row: int) -> int:
        if not self.is_inside(col, row):
            raise IndexError(f'Grid cell out of bounds: ({col}, {row})')
        return self.data[self.index(col, row)]


def from_pgm_map(
    pgm_map: PgmMap,
    occupied_cost: int = 100,
    free_cost: int = 0,
    unknown_cost: int = -1,
) -> Costmap2D:
    metadata = pgm_map.metadata
    values = []

    for row in range(metadata.height):
        for col in range(metadata.width):
            pixel = pgm_map.pixel_at_grid(col, row)
            occupancy = _pixel_to_occupancy_probability(pixel, metadata.negate)

            if occupancy > metadata.occupied_thresh:
                values.append(occupied_cost)
            elif occupancy < metadata.free_thresh:
                values.append(free_cost)
            else:
                values.append(unknown_cost)

    return Costmap2D(
        width=metadata.width,
        height=metadata.height,
        resolution=metadata.resolution,
        origin=metadata.origin,
        data=values,
        free_cost=free_cost,
        occupied_cost=occupied_cost,
        unknown_cost=unknown_cost,
    )


def _pixel_to_occupancy_probability(pixel: int, negate: int) -> float:
    if negate:
        return pixel / 255.0
    return (255 - pixel) / 255.0
