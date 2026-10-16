"""Run the connectome-driven fly on a map without the web page.

    python navigate/run.py navigate/maps/maze.json --seed 0 --max-s 90
    python navigate/run.py my_map.json --out trajectory.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flyconn  # noqa: E402
from flyconn.agent import FlyAgent, FlyConfig  # noqa: E402
from flyconn.world import World  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Run the fly on a map file.')
    ap.add_argument('map', type=Path, help='map JSON with a "rows" list (# . S G)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-s', type=float, default=90.0, help='simulated seconds before giving up')
    ap.add_argument('--see-through-walls', action='store_true')
    ap.add_argument('--out', type=Path, help='write every frame to this JSON file')
    args = ap.parse_args()

    world = World.from_dict(json.loads(args.map.read_text()))
    cfg = FlyConfig(seed=args.seed, see_through_walls=args.see_through_walls)
    print('loading connectome...', flush=True)
    agent = FlyAgent(flyconn.load(), config=cfg)
    agent.calibrate()

    t0, last = time.time(), [0.0]

    def progress(f):
        if f['t_ms'] - last[0] >= 1000:
            last[0] = f['t_ms']
            print(f"  t={f['t_ms'] / 1000:5.1f}s  goal {f['goal_dist_mm']:5.1f} mm  "
                  f"{'in sight' if f['goal_visible'] else 'hidden  '}  steer {f['steer']:+.2f}", flush=True)

    frames = agent.run(world, max_s=args.max_s, callback=progress)
    f = frames[-1]
    verdict = 'REACHED the goal' if f['reached'] else 'did not reach the goal'
    print(f"{verdict} after {f['t_ms'] / 1000:.1f} simulated s "
          f"({time.time() - t0:.0f} s of compute)")
    if args.out:
        args.out.write_text(json.dumps({'map': world.name, 'frames': frames}))
        print(f'frames written to {args.out}')


if __name__ == '__main__':
    main()
