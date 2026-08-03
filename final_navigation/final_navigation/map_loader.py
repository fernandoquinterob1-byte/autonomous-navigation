import os
from dataclasses import dataclass
from typing import List, Tuple

from ament_index_python.packages import get_package_share_directory
import yaml


@dataclass(frozen=True)
class MapMetadata:
    yaml_path: str
    image_path: str
    width: int
    height: int
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float


class PgmMap:
    def __init__(self, metadata: MapMetadata, pixels_top_down: List[int]):
        self.metadata = metadata
        self.pixels_top_down = pixels_top_down

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    @property
    def resolution(self) -> float:
        return self.metadata.resolution

    @property
    def origin(self) -> Tuple[float, float, float]:
        return self.metadata.origin

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        origin_x, origin_y, _ = self.origin
        col = int((x - origin_x) / self.resolution)
        row = int((y - origin_y) / self.resolution)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        origin_x, origin_y, _ = self.origin
        x = origin_x + (col + 0.5) * self.resolution
        y = origin_y + (row + 0.5) * self.resolution
        return x, y

    def is_inside_grid(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def pixel_at_grid(self, col: int, row: int) -> int:
        if not self.is_inside_grid(col, row):
            raise IndexError(f'Grid cell out of bounds: ({col}, {row})')
        image_row = self.height - 1 - row
        return self.pixels_top_down[image_row * self.width + col]


def load_map(yaml_path: str) -> PgmMap:
    yaml_path = _resolve_map_path(yaml_path)
    with open(yaml_path, 'r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)

    image_name = data['image']
    image_path = image_name
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(yaml_path), image_name)

    width, height, pixels = _read_pgm(image_path)
    origin = tuple(float(value) for value in data['origin'])
    metadata = MapMetadata(
        yaml_path=yaml_path,
        image_path=os.path.abspath(image_path),
        width=width,
        height=height,
        resolution=float(data['resolution']),
        origin=(origin[0], origin[1], origin[2]),
        negate=int(data.get('negate', 0)),
        occupied_thresh=float(data['occupied_thresh']),
        free_thresh=float(data['free_thresh']),
    )
    return PgmMap(metadata, pixels)


def _resolve_map_path(path: str) -> str:
    expanded_path = os.path.expanduser(path)
    package_prefix = 'package://'

    if not expanded_path.startswith(package_prefix):
        return os.path.abspath(expanded_path)

    package_path = expanded_path[len(package_prefix):]
    package_name, separator, relative_path = package_path.partition('/')
    if not package_name or not separator or not relative_path:
        raise ValueError(
            'Package map paths must use package://<package>/<path>')

    return os.path.join(
        get_package_share_directory(package_name),
        relative_path,
    )


def _read_pgm(path: str) -> Tuple[int, int, List[int]]:
    with open(path, 'rb') as stream:
        magic = _read_token(stream)
        if magic not in (b'P5', b'P2'):
            raise ValueError(f'Unsupported PGM format {magic!r} in {path}')

        width = int(_read_token(stream))
        height = int(_read_token(stream))
        max_value = int(_read_token(stream))
        if max_value <= 0 or max_value > 255:
            raise ValueError(f'Unsupported PGM max value {max_value} in {path}')

        pixel_count = width * height
        if magic == b'P5':
            raw = stream.read(pixel_count)
            if len(raw) != pixel_count:
                raise ValueError(
                    f'Expected {pixel_count} pixels in {path}, got {len(raw)}')
            return width, height, list(raw)

        pixels = [int(_read_token(stream)) for _ in range(pixel_count)]
        return width, height, pixels


def _read_token(stream) -> bytes:
    token = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            if token:
                return bytes(token)
            raise EOFError('Unexpected end of PGM file')
        if char == b'#':
            stream.readline()
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)
