"""Whole-brain leaky integrate-and-fire (LIF) model on the FlyWire connectome.

Same equations and parameters as Shiu et al. (Nature 2024), the model used by
eonsystemspbc/fly-brain, compiled with Numba so it runs fast on a CPU:

    dv/dt = (v_0 - v + g) / t_mbr          membrane potential  (frozen while refractory)
    dg/dt = -g / tau                       synaptic input      (frozen while refractory)
    spike when v > v_th  ->  v = v_rst, g = 0, refractory for t_rfc
    a spike adds w_syn * (signed synapse count) to every target's g,
    t_dly later (dropped if the target is refractory).

The linear ODEs are integrated exactly over each time step, as Brian2's
'linear' method does. Driven neurons fire as Poisson processes at a chosen
rate with no refractory period, like the optogenetic activation (a 250x
synaptic kick) in the original paper.

The simulation is event-driven: a neuron at rest (v = v_0, g = 0) stays at
rest until input arrives, so only neurons away from rest are updated. Neurons
within ``rest_tol`` mV of rest are snapped back to it (0.01 mV, against a
7 mV distance to threshold).
"""

import numpy as np
import scipy.sparse as sp
from numba import njit

PARAMS = {
    'dt': 0.1,        # ms
    'v_0': -52.0,     # mV resting potential
    'v_rst': -52.0,   # mV reset potential
    'v_th': -45.0,    # mV spike threshold
    't_mbr': 20.0,    # ms membrane time constant
    'tau': 5.0,       # ms synaptic time constant
    't_rfc': 2.2,     # ms refractory period
    't_dly': 1.8,     # ms synaptic delay
    'w_syn': 0.275,   # mV per synapse
    'rest_tol': 1e-2,  # mV
}


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _run(n_steps, t0, v, g, ref, ref_len, indptr, indices, data,
         drive_idx, drive_p, qbuf, qlen, active, n_active, is_active,
         last_spike, counts, v0, v_th, v_rst, ev, eg, k_gv, tol, delay):
    for step in range(n_steps):
        t = t0 + step
        slot = t % delay
        # 1. deliver spikes emitted t_dly ago
        for k in range(qlen[slot]):
            pre = qbuf[slot, k]
            for j in range(indptr[pre], indptr[pre + 1]):
                post = indices[j]
                if ref[post] == 0:
                    g[post] += data[j]
                    if not is_active[post]:
                        is_active[post] = True
                        active[n_active] = post
                        n_active += 1
        n_spk = 0
        # 2. integrate active neurons, detect spikes, retire neurons at rest
        i = 0
        while i < n_active:
            nid = active[i]
            if ref[nid] > 0:
                ref[nid] -= 1
                i += 1
                continue
            vv = v0 + (v[nid] - v0) * ev + g[nid] * k_gv
            g[nid] *= eg
            if vv > v_th:
                v[nid] = v_rst
                g[nid] = 0.0
                ref[nid] = ref_len[nid]
                last_spike[nid] = t
                qbuf[slot, n_spk] = nid
                n_spk += 1
                counts[nid] += 1
                i += 1
            elif abs(vv - v0) < tol and abs(g[nid]) < tol:
                v[nid] = v0
                g[nid] = 0.0
                is_active[nid] = False
                n_active -= 1
                active[i] = active[n_active]
            else:
                v[nid] = vv
                i += 1
        # 3. Poisson-driven neurons
        for k in range(drive_idx.shape[0]):
            nid = drive_idx[k]
            if last_spike[nid] != t and np.random.random() < drive_p[k]:
                v[nid] = v_rst
                g[nid] = 0.0
                ref[nid] = ref_len[nid]
                last_spike[nid] = t
                qbuf[slot, n_spk] = nid
                n_spk += 1
                counts[nid] += 1
                if ref[nid] > 0 and not is_active[nid]:
                    is_active[nid] = True
                    active[n_active] = nid
                    n_active += 1
        qlen[slot] = n_spk
    return n_active


class Brain:
    """Spiking simulation of every neuron in the connectome."""

    def __init__(self, connectome, params=None, seed=0):
        self.p = {**PARAMS, **(params or {})}
        p = self.p
        self.n = connectome.n
        W = sp.csr_matrix(connectome.matrix(signed=True).astype(np.float32) * p['w_syn'])
        self.indptr = W.indptr.astype(np.int64)
        self.indices = W.indices.astype(np.int32)
        self.data = W.data.astype(np.float32)
        self.delay = int(round(p['t_dly'] / p['dt']))
        self.ref_steps = int(round(p['t_rfc'] / p['dt']))
        # exact one-step propagators for the (v, g) linear system
        ev, eg = np.exp(-p['dt'] / p['t_mbr']), np.exp(-p['dt'] / p['tau'])
        self.ev, self.eg = float(ev), float(eg)
        self.k_gv = float((eg - ev) / (1 - p['t_mbr'] / p['tau']))
        self.seed = seed
        self.reset()

    def reset(self):
        p = self.p
        n = self.n
        self.v = np.full(n, p['v_0'], np.float32)
        self.g = np.zeros(n, np.float32)
        self.ref = np.zeros(n, np.int16)
        self.ref_len = np.full(n, self.ref_steps, np.int16)
        self.qbuf = np.zeros((self.delay, n), np.int32)
        self.qlen = np.zeros(self.delay, np.int64)
        self.active = np.zeros(n, np.int32)
        self.n_active = 0
        self.is_active = np.zeros(n, np.bool_)
        self.last_spike = np.full(n, -1, np.int64)
        self.t = 0
        self.drive_idx = np.empty(0, np.int32)
        self.drive_p = np.empty(0, np.float64)
        _seed(self.seed)

    def set_drive(self, idx, rates_hz):
        """Poisson-drive neurons ``idx`` at ``rates_hz`` (scalar or per neuron).
        Replaces any previous drive."""
        idx = np.asarray(idx, np.int64).ravel()
        rates = np.broadcast_to(np.asarray(rates_hz, np.float64), idx.shape)
        keep = rates > 0
        self.ref_len[self.drive_idx] = self.ref_steps
        self.drive_idx = idx[keep].astype(np.int32)
        self.drive_p = rates[keep] * self.p['dt'] / 1000.0
        self.ref_len[self.drive_idx] = 0  # as in the original model

    def run(self, ms, record=None):
        """Run for ``ms`` milliseconds. Returns spike counts in that window for
        the neurons in ``record`` (or for every neuron if ``record`` is None)."""
        p = self.p
        n_steps = int(round(ms / p['dt']))
        counts = np.zeros(self.n, np.int32)
        self.n_active = _run(
            n_steps, self.t, self.v, self.g, self.ref, self.ref_len,
            self.indptr, self.indices, self.data, self.drive_idx, self.drive_p,
            self.qbuf, self.qlen, self.active, self.n_active, self.is_active,
            self.last_spike, counts, p['v_0'], p['v_th'], p['v_rst'],
            self.ev, self.eg, self.k_gv, p['rest_tol'], self.delay)
        self.t += n_steps
        return counts if record is None else counts[np.asarray(record)]
