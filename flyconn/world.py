"""A 2D grid world: walls, a start and a goal, plus line-of-sight geometry.

Maps are written as rows of characters:

    '#'  wall        '.'  floor        'S'  start        'G'  goal

Coordinates are in millimetres with x to the right and y *down* (screen
convention). Angles are in radians; a positive relative angle means the
point is to the fly's left.
"""

from dataclasses import dataclass, field

import numpy as np

CELL_MM = 2.0  # one grid cell is 2 x 2 mm (a fruit fly is ~2.5 mm long)


@dataclass
class World:
    grid: np.ndarray            # bool, True = wall, shape (rows, cols)
    start: tuple                # (x_mm, y_mm)
    goal: tuple                 # (x_mm, y_mm)
    cell_mm: float = CELL_MM
    start_heading: float = 0.0  # radians
    name: str = ''
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows, cell_mm=CELL_MM, start_heading_deg=0.0, name=''):
        width = max(len(r) for r in rows)
        rows = [r.ljust(width, '.') for r in rows]
        grid = np.array([[ch == '#' for ch in r] for r in rows])
        start = goal = None
        for i, r in enumerate(rows):
            for j, ch in enumerate(r):
                centre = ((j + 0.5) * cell_mm, (i + 0.5) * cell_mm)
                if ch == 'S':
                    start = centre
                elif ch == 'G':
                    goal = centre
        if start is None or goal is None:
            raise ValueError("map needs one 'S' (start) and one 'G' (goal)")
        return cls(grid=grid, start=start, goal=goal, cell_mm=cell_mm,
                   start_heading=np.radians(start_heading_deg), name=name)

    @classmethod
    def from_dict(cls, d):
        return cls.from_rows(d['rows'], d.get('cell_mm', CELL_MM),
                             d.get('start_heading_deg', 0.0), d.get('name', ''))

    @property
    def size_mm(self):
        return self.grid.shape[1] * self.cell_mm, self.grid.shape[0] * self.cell_mm

    def is_wall(self, x, y):
        """True for wall cells and for anything outside the map."""
        j, i = int(np.floor(x / self.cell_mm)), int(np.floor(y / self.cell_mm))
        if i < 0 or j < 0 or i >= self.grid.shape[0] or j >= self.grid.shape[1]:
            return True
        return bool(self.grid[i, j])

    def blocked(self, x, y, radius):
        """True if a disc of ``radius`` mm at (x, y) overlaps a wall."""
        for dx in (-radius, 0.0, radius):
            for dy in (-radius, 0.0, radius):
                if self.is_wall(x + dx, y + dy):
                    return True
        return False

    def ray(self, x, y, angle, max_mm):
        """Distance (mm) from (x, y) along ``angle`` to the first wall,
        capped at ``max_mm``. Grid traversal (Amanatides & Woo)."""
        c = self.cell_mm
        dx, dy = np.cos(angle), np.sin(angle)
        i, j = int(np.floor(y / c)), int(np.floor(x / c))
        step_j, step_i = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
        t_max_x = (((j + (dx > 0)) * c - x) / dx) if dx else np.inf
        t_max_y = (((i + (dy > 0)) * c - y) / dy) if dy else np.inf
        t_dx = abs(c / dx) if dx else np.inf
        t_dy = abs(c / dy) if dy else np.inf
        t = 0.0
        rows, cols = self.grid.shape
        while t < max_mm:
            if t_max_x < t_max_y:
                t, t_max_x, j = t_max_x, t_max_x + t_dx, j + step_j
            else:
                t, t_max_y, i = t_max_y, t_max_y + t_dy, i + step_i
            if i < 0 or j < 0 or i >= rows or j >= cols or self.grid[i, j]:
                return min(t, max_mm)
        return max_mm

    def line_of_sight(self, a, b):
        """True if no wall lies on the straight line from a to b."""
        d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if d == 0:
            return True
        angle = np.arctan2(b[1] - a[1], b[0] - a[0])
        return self.ray(a[0], a[1], angle, d) >= d - 1e-6


def wrap(angle):
    """Wrap an angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi
