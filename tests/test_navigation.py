"""Tests for the spiking brain engine, the grid world and the fly's senses,
on tiny hand-built inputs (no real data needed)."""

import numpy as np
import pytest

from flyconn.brain import Brain
from flyconn.world import World, wrap
from tests.test_analysis import make


# ---------------------------------------------------------------- brain

def test_quiet_brain_stays_silent():
    c = make([(0, 1, 50, 1)], 3)
    b = Brain(c)
    assert b.run(200).sum() == 0
    assert b.n_active == 0


def test_poisson_drive_rate():
    c = make([(0, 1, 1, 1)], 3)
    b = Brain(c, seed=1)
    b.set_drive([0], 100.0)
    rate = b.run(2000)[0] / 2.0
    assert 85 <= rate <= 115


def test_strong_excitation_propagates_and_inhibition_blocks():
    # 0 -> 1 strongly excitatory; 2 -> 1 strongly inhibitory; 0 -> 3 excitatory
    c = make([(0, 1, 100, 1), (0, 3, 100, 1), (2, 1, 300, -1)], 4)
    b = Brain(c, seed=0)
    b.set_drive([0], 100.0)
    counts = b.run(1000)
    assert counts[1] > 20 and counts[3] > 20
    b.reset()
    b.set_drive([0, 2], 100.0)
    counts = b.run(1000)
    assert counts[1] < counts[3] / 4


def test_weak_input_stays_subthreshold():
    # one synapse = 0.275 mV of input: far below the 7 mV to threshold
    c = make([(0, 1, 1, 1)], 2)
    b = Brain(c)
    b.set_drive([0], 50.0)
    assert b.run(1000)[1] == 0


def test_reset_is_reproducible():
    c = make([(0, 1, 100, 1)], 2)
    b = Brain(c, seed=3)
    b.set_drive([0], 80.0)
    first = b.run(500)
    b.reset()
    b.set_drive([0], 80.0)
    assert np.array_equal(first, b.run(500))


# ---------------------------------------------------------------- world

ROWS = [
    '#######',
    '#S...G#',
    '#.....#',
    '#######',
]


def test_world_parses_start_goal_and_walls():
    w = World.from_rows(ROWS)
    assert w.start == (3.0, 3.0) and w.goal == (11.0, 3.0)
    assert w.is_wall(1.0, 1.0) and not w.is_wall(5.0, 3.0)
    assert w.is_wall(-1, 3) and w.is_wall(100, 3)   # outside counts as wall


def test_world_requires_start_and_goal():
    with pytest.raises(ValueError):
        World.from_rows(['#...#'])


def test_ray_distance_to_wall():
    w = World.from_rows(ROWS)
    # from the start centre (3, 3) looking right: wall cell begins at x = 12
    assert w.ray(3.0, 3.0, 0.0, 50.0) == pytest.approx(9.0)
    # looking up: wall row ends at y = 2
    assert w.ray(3.0, 3.0, -np.pi / 2, 50.0) == pytest.approx(1.0)
    assert w.ray(3.0, 3.0, 0.0, 4.0) == 4.0  # capped


def test_line_of_sight_blocked_by_wall():
    w = World.from_rows(['#######', '#S.#.G#', '#######'])
    assert not w.line_of_sight(w.start, w.goal)
    w2 = World.from_rows(ROWS)
    assert w2.line_of_sight(w2.start, w2.goal)


def test_wrap():
    assert wrap(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    assert wrap(-3 * np.pi / 2) == pytest.approx(np.pi / 2)


# ---------------------------------------------------------------- senses

def tiny_fly():
    from flyconn.agent import FlyAgent
    groups = [(t, s) for t in ['LC10a', 'LPLC2', 'DNa01', 'DNa02', 'DNp09', 'DNp01', 'MDN']
              for s in ['left', 'right']]
    c = make([(0, 1, 1, 1)], len(groups),
             side=[s for _, s in groups], cell_type=[t for t, _ in groups])
    return FlyAgent(c)


def test_goal_on_left_drives_left_lc10a():
    fly = tiny_fly()
    # facing up (heading -90 deg, y down); goal to the west is on the fly's left
    w = World.from_rows(['#########', '#G.....S#', '#.......#', '#########'], start_heading_deg=-90)
    fly.reset(w)
    rates, seen = fly.sense()
    assert seen['goal_visible'] and seen['goal_angle_deg'] == pytest.approx(90)
    assert rates['LC10a_L'] > 0 and rates['LC10a_R'] == 0


def test_wall_on_right_drives_right_lplc2():
    fly = tiny_fly()
    # facing up along the east wall: the wall is on the fly's right
    w = World.from_rows(['#####', '#..G#', '#...#', '#..S#', '#####'], start_heading_deg=-90)
    fly.reset(w)
    rates, _ = fly.sense()
    assert rates['LPLC2_R'] > rates['LPLC2_L']


def test_steering_sign():
    fly = tiny_fly()
    fly.dn_peak = {k: 10.0 for k in fly.STEER_L + fly.STEER_R}
    left = {'DNa01_L': 10, 'DNa02_L': 10, 'DNa01_R': 0, 'DNa02_R': 0}
    assert fly.steering(left) == pytest.approx(2.0)   # + = turn left
