"""Local web app: draw a map, then watch the connectome-driven fly try to
reach the goal.

    python navigate/server.py            # then open http://localhost:8000

The brain runs here in Python; the page polls for new frames. Standard
library only (no web framework).
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import flyconn  # noqa: E402
from flyconn.agent import FlyAgent, FlyConfig  # noqa: E402
from flyconn.world import World  # noqa: E402

MAPS = ROOT / 'maps'
MAX_CELLS = 80


class Simulation:
    """One fly, one run at a time, stepped on a background thread."""

    def __init__(self, agent):
        self.agent = agent
        self.lock = threading.Lock()
        self.thread = None
        self.stop_flag = threading.Event()
        self.run_id = 0
        self.frames = []
        self.status = 'idle'
        self.error = None

    def start(self, world, settings):
        self.stop()
        cfg = self.agent.cfg
        cfg.see_through_walls = bool(settings.get('see_through_walls', False))
        cfg.seed = int(settings.get('seed', 0))
        cfg.speed_mm_s = float(settings.get('speed_mm_s', 12.0))
        cfg.noise_deg_s = float(settings.get('noise_deg_s', 40.0))
        self.agent.brain.seed = cfg.seed
        max_s = float(settings.get('max_s', 60.0))
        with self.lock:
            self.run_id += 1
            self.frames = []
            self.status = 'running'
            self.error = None
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._loop, args=(world, max_s, self.run_id), daemon=True)
        self.thread.start()
        return self.run_id

    def _loop(self, world, max_s, run_id):
        try:
            self.agent.reset(world)
            start = time.time()
            while not self.stop_flag.is_set():
                f = self.agent.step()
                f['wall_s'] = round(time.time() - start, 2)
                with self.lock:
                    if run_id != self.run_id:
                        return
                    self.frames.append(f)
                    if f['reached']:
                        self.status = 'reached'
                        return
                    if f['t_ms'] >= max_s * 1000:
                        self.status = 'timeout'
                        return
            with self.lock:
                self.status = 'stopped'
        except Exception as exc:  # report to the page instead of dying silently
            with self.lock:
                self.status, self.error = 'error', repr(exc)

    def stop(self):
        self.stop_flag.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def poll(self, run_id, since):
        with self.lock:
            if run_id != self.run_id:
                return {'run_id': self.run_id, 'frames': [], 'status': 'superseded'}
            return {'run_id': run_id, 'frames': self.frames[since:], 'status': self.status,
                    'error': self.error}


def load_presets():
    out = []
    for p in sorted(MAPS.glob('*.json')):
        d = json.loads(p.read_text())
        d['id'] = p.stem
        out.append(d)
    return out


def make_handler(sim, info):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            pass

        def _send(self, code, body, ctype='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _json_body(self):
            n = int(self.headers.get('Content-Length', 0))
            if n > 1_000_000:
                raise ValueError('request too large')
            return json.loads(self.rfile.read(n) or b'{}')

        def do_GET(self):
            path, _, query = self.path.partition('?')
            if path in ('/', '/index.html'):
                return self._send(200, (ROOT / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/info':
                return self._send(200, {**info, 'presets': load_presets()})
            if path == '/api/frames':
                q = dict(kv.split('=', 1) for kv in query.split('&') if '=' in kv)
                return self._send(200, sim.poll(int(q.get('run', 0)), int(q.get('since', 0))))
            self._send(404, {'error': 'not found'})

        def do_POST(self):
            try:
                body = self._json_body()
                if self.path == '/api/run':
                    rows = body['rows']
                    if not rows or len(rows) > MAX_CELLS or max(len(r) for r in rows) > MAX_CELLS:
                        raise ValueError(f'map must be 1-{MAX_CELLS} cells in each direction')
                    world = World.from_rows(rows, start_heading_deg=float(body.get('start_heading_deg', 0)))
                    run_id = sim.start(world, body.get('settings', {}))
                    w, h = world.size_mm
                    return self._send(200, {'run_id': run_id, 'width_mm': w, 'height_mm': h,
                                            'cell_mm': world.cell_mm})
                if self.path == '/api/stop':
                    sim.stop()
                    return self._send(200, {'ok': True})
                self._send(404, {'error': 'not found'})
            except (KeyError, ValueError) as exc:
                self._send(400, {'error': str(exc)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()

    print('Loading the FlyWire connectome (138,639 neurons)...', flush=True)
    t = time.time()
    agent = FlyAgent(flyconn.load(), config=FlyConfig())
    print(f'  loaded in {time.time() - t:.1f}s; calibrating steering neurons...', flush=True)
    peaks = agent.calibrate()
    print('  peak rates (Hz): ' + ', '.join(f'{k} {v:.0f}' for k, v in peaks.items()), flush=True)
    info = {
        'neurons': agent.brain.n,
        'calibration_hz': peaks,
        'inputs': {k: len(v) for k, v in agent.inputs.items()},
        'outputs': {k: len(v) for k, v in agent.outputs.items()},
        'tick_ms': agent.cfg.tick_ms,
        'cell_mm': 2.0,
    }
    sim = Simulation(agent)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(sim, info))
    print(f'Ready: open http://localhost:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()


if __name__ == '__main__':
    main()
