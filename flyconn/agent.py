"""A virtual fly whose steering comes from the whole-brain connectome model.

Closed loop, repeated every ``tick_ms`` of simulated time:

    world --eyes--> visual neurons --connectome (138,639 LIF neurons)--> descending neurons --body--> movement

What is the brain and what is the body
--------------------------------------
* Senses (model): the goal is a visual object. When it is in the left half
  of the visual field it drives the left LC10a neurons (the visual projection
  neurons flies use to track objects), more strongly the further it sits from
  straight ahead; likewise on the right. Walls close to an eye loom, which
  drives that side's LPLC2 looming detectors.
* Brain (connectome, not tuned): those rates are injected as Poisson spikes
  and the full LIF network runs. No weights are fitted.
* Readout (literature): steering is the left-minus-right activity of the
  steering descending neurons DNa01 and DNa02 (Rayshubskiy et al. 2020);
  more activity on one side turns the fly towards that side. Each neuron's
  rate is read as a fraction of its maximum, measured once at start-up by
  ``calibrate`` (the left DNa02 is ~4x more excitable than the right one in
  the reconstruction, so raw rates would bias the fly to the left).
* Body (model): walks forward at a set speed, slows down in sharp turns,
  adds small random heading noise (exploration), and slides along walls.

In the connectome, LC10a drives the *same-side* DNa02 (turn towards a target)
while LPLC2 drives the *opposite-side* DNa01/DNa02 (turn away from a looming
wall), so one readout gives both behaviours.
"""

from dataclasses import dataclass, field

import numpy as np

from .brain import Brain
from .world import wrap

DEG = np.pi / 180


@dataclass
class FlyConfig:
    tick_ms: float = 20.0          # brain time simulated per control step
    speed_mm_s: float = 12.0       # walking speed
    body_radius_mm: float = 0.9
    # goal -> LC10a
    goal_rate_hz: float = 60.0     # peak LC10a drive
    full_drive_deg: float = 30.0   # goal this far off-centre gives full drive
    see_through_walls: bool = False
    # walls -> LPLC2
    wall_rate_hz: float = 90.0     # LPLC2 drive when a wall is at the body
    wall_range_mm: float = 8.0
    wall_rays_deg: tuple = (10, 30, 50, 70)
    # descending neurons -> turning
    dn_smooth_ms: float = 100.0    # motor integration time constant
    turn_gain_deg_s: float = 200.0  # per unit of normalised steering
    max_turn_deg_s: float = 360.0
    noise_deg_s: float = 40.0      # random heading diffusion (deg / sqrt(s))
    goal_radius_mm: float = 3.0
    seed: int = 0


@dataclass
class FlyState:
    x: float
    y: float
    heading: float
    t_ms: float = 0.0
    dn_rate: dict = field(default_factory=dict)   # smoothed Hz per readout group
    reached: bool = False


class FlyAgent:
    # neuron groups: name -> (cell type, side)
    INPUTS = {
        'LC10a_L': ('LC10a', 'left'), 'LC10a_R': ('LC10a', 'right'),
        'LPLC2_L': ('LPLC2', 'left'), 'LPLC2_R': ('LPLC2', 'right'),
    }
    OUTPUTS = {
        'DNa01_L': ('DNa01', 'left'), 'DNa01_R': ('DNa01', 'right'),
        'DNa02_L': ('DNa02', 'left'), 'DNa02_R': ('DNa02', 'right'),
        # recorded for display only
        'P9_L': ('DNp09', 'left'), 'P9_R': ('DNp09', 'right'),
        'GF_L': ('DNp01', 'left'), 'GF_R': ('DNp01', 'right'),
        'MDN_L': ('MDN', 'left'), 'MDN_R': ('MDN', 'right'),
    }

    def __init__(self, connectome, brain=None, config=None):
        self.cfg = config or FlyConfig()
        self.brain = brain or Brain(connectome, seed=self.cfg.seed)
        nt = connectome.neurons
        ct = nt['cell_type'].astype(str).to_numpy()
        side = nt['side'].astype(str).to_numpy()
        pick = lambda t, s: np.flatnonzero((ct == t) & (side == s))  # noqa: E731
        self.inputs = {k: pick(*v) for k, v in self.INPUTS.items()}
        self.outputs = {k: pick(*v) for k, v in self.OUTPUTS.items()}
        for k, v in {**self.inputs, **self.outputs}.items():
            if not len(v):
                raise ValueError(f'no neurons found for {k}')
        self._in_idx = np.concatenate(list(self.inputs.values()))
        self._out_idx = np.concatenate(list(self.outputs.values()))
        self.rng = np.random.default_rng(self.cfg.seed)

    STEER_L = ('DNa01_L', 'DNa02_L')
    STEER_R = ('DNa01_R', 'DNa02_R')

    def calibrate(self, rate_hz=60.0, ms=400.0):
        """Drive each sensory input group alone and record every steering
        neuron's peak rate; the readout divides by these peaks."""
        peak = {k: 0.0 for k in self.STEER_L + self.STEER_R}
        for group in self.inputs.values():
            self.brain.reset()
            self.brain.set_drive(group, rate_hz)
            self.brain.run(100)
            counts = self.brain.run(ms)
            for k in peak:
                hz = counts[self.outputs[k]].mean() * 1000.0 / ms
                peak[k] = max(peak[k], hz)
        self.brain.reset()
        self.dn_peak = {k: max(float(v), 1.0) for k, v in peak.items()}
        return self.dn_peak

    def steering(self, rates):
        """Normalised left-minus-right steering signal (+ = turn left)."""
        if not hasattr(self, 'dn_peak'):
            self.calibrate()
        side = lambda keys: sum(rates[k] / self.dn_peak[k] for k in keys)  # noqa: E731
        return side(self.STEER_L) - side(self.STEER_R)

    # ------------------------------------------------------------------
    def reset(self, world):
        self.world = world
        self.brain.reset()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.state = FlyState(*world.start, heading=world.start_heading,
                              dn_rate={k: 0.0 for k in self.outputs})
        return self.state

    def sense(self):
        """Firing rates (Hz) for each input group, plus what the fly sees."""
        c, s, w = self.cfg, self.state, self.world
        gx, gy = w.goal
        dist = float(np.hypot(gx - s.x, gy - s.y))
        visible = c.see_through_walls or w.line_of_sight((s.x, s.y), (gx, gy))
        rel = wrap(s.heading - np.arctan2(gy - s.y, gx - s.x))  # + = left
        size = min(1.0, 0.5 + 10.0 / max(dist, 1e-6))           # nearer = bigger
        drive = c.goal_rate_hz * size * min(1.0, abs(rel) / (c.full_drive_deg * DEG)) if visible else 0.0
        goal_l, goal_r = (drive, 0.0) if rel > 0 else (0.0, drive)

        def loom(sign):
            d = min(w.ray(s.x, s.y, s.heading - sign * a * DEG, c.wall_range_mm)
                    for a in c.wall_rays_deg)
            closeness = (c.wall_range_mm - d) / (c.wall_range_mm - c.body_radius_mm)
            return c.wall_rate_hz * float(np.clip(closeness, 0, 1)), d

        wall_l, dl = loom(+1)   # rays on the left (heading - angle)
        wall_r, dr = loom(-1)
        rates = {'LC10a_L': goal_l, 'LC10a_R': goal_r, 'LPLC2_L': wall_l, 'LPLC2_R': wall_r}
        seen = {'goal_visible': bool(visible), 'goal_angle_deg': float(rel / DEG),
                'goal_dist_mm': dist, 'wall_left_mm': dl, 'wall_right_mm': dr}
        return rates, seen

    def step(self):
        """Advance one control tick. Returns a dict describing the tick."""
        c, s = self.cfg, self.state
        rates, seen = self.sense()
        self.brain.set_drive(self._in_idx, np.concatenate(
            [np.full(len(self.inputs[k]), rates[k]) for k in self.inputs]))
        counts = self.brain.run(c.tick_ms)
        raw = {}
        for k, idx in self.outputs.items():
            raw[k] = counts[idx].sum() / len(idx) * 1000.0 / c.tick_ms  # Hz per neuron
        a = c.tick_ms / c.dn_smooth_ms
        for k in raw:
            s.dn_rate[k] += a * (raw[k] - s.dn_rate[k])
        steer = self.steering(s.dn_rate)
        turn = np.clip(c.turn_gain_deg_s * steer, -c.max_turn_deg_s, c.max_turn_deg_s)
        dt = c.tick_ms / 1000.0
        noise = self.rng.normal(0, c.noise_deg_s * np.sqrt(dt))
        s.heading = wrap(s.heading - (turn * dt + noise) * DEG)  # left turn = heading decreases
        speed = c.speed_mm_s * (1 - 0.5 * min(1.0, abs(turn) / c.max_turn_deg_s))
        self._move(speed * dt)
        s.t_ms += c.tick_ms
        s.reached = seen['goal_dist_mm'] <= c.goal_radius_mm
        return {
            't_ms': s.t_ms, 'x': s.x, 'y': s.y, 'heading_deg': float(s.heading / DEG),
            'turn_deg_s': float(turn), 'steer': round(float(steer), 3), 'speed_mm_s': float(speed), 'reached': s.reached,
            'inputs_hz': {k: round(float(v), 1) for k, v in rates.items()},
            'dn_hz': {k: round(float(v), 1) for k, v in s.dn_rate.items()},
            'active_neurons': int(self.brain.n_active),
            'spikes': int(counts.sum()),
            **seen,
        }

    def _move(self, dist):
        s, w, rad = self.state, self.world, self.cfg.body_radius_mm
        nx, ny = s.x + dist * np.cos(s.heading), s.y + dist * np.sin(s.heading)
        if not w.blocked(nx, ny, rad):
            s.x, s.y = nx, ny
        elif not w.blocked(nx, s.y, rad):     # slide along a vertical wall
            s.x = nx
        elif not w.blocked(s.x, ny, rad):     # slide along a horizontal wall
            s.y = ny

    def run(self, world, max_s=60.0, callback=None):
        """Run until the goal is reached or ``max_s`` simulated seconds pass."""
        self.reset(world)
        frames = []
        while self.state.t_ms < max_s * 1000:
            f = self.step()
            frames.append(f)
            if callback:
                callback(f)
            if f['reached']:
                break
        return frames
