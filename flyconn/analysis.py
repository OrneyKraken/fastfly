"""Graph analyses of the FlyWire connectome."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

# Threshold used for "strong" connections, following FlyWire practice
# (Dorkenwald et al. 2024): >= 5 synapses filters out most false positives.
STRONG = 5

SENSORY_MODALITIES = ['mechanosensory', 'gustatory', 'olfactory', 'visual']


# ---------------------------------------------------------------------------
# Global statistics
# ---------------------------------------------------------------------------

def summary(c):
    """Headline numbers for the whole brain."""
    e = c.edges
    strong = e[e['synapses'] >= STRONG]
    W = c.matrix(STRONG)
    B = (W > 0).astype(np.int8)
    reciprocal = B.multiply(B.T).nnz  # each reciprocal pair counted twice
    density = strong.shape[0] / (c.n * (c.n - 1))
    return {
        'neurons': int(c.n),
        'connections': int(len(e)),
        'synapses': int(e['synapses'].sum()),
        'strong_connections': int(len(strong)),
        'excitatory_synapse_frac': float(e.loc[e['sign'] > 0, 'synapses'].sum() / e['synapses'].sum()),
        'cell_types': int(c.neurons['cell_type'].nunique() - 1),  # minus 'unknown'
        'median_partners_out': float(np.median(np.bincount(e['pre'], minlength=c.n))),
        'median_partners_in': float(np.median(np.bincount(e['post'], minlength=c.n))),
        'max_synapses_one_connection': int(e['synapses'].max()),
        'strong_reciprocity': float(reciprocal / len(strong)),
        'strong_density': float(density),
        # how much more often A<->B occurs than if strong edges were random
        'reciprocity_enrichment': float(reciprocal / len(strong) / density),
    }


def neuron_stats(c):
    """Per-neuron degree, synapse totals and PageRank."""
    e = c.edges
    stats = c.neurons[['root_id', 'super_class', 'cell_class', 'cell_type',
                       'top_nt', 'side']].copy()
    stats['out_partners'] = np.bincount(e['pre'], minlength=c.n)
    stats['in_partners'] = np.bincount(e['post'], minlength=c.n)
    stats['out_synapses'] = np.bincount(e['pre'], weights=e['synapses'], minlength=c.n).astype(int)
    stats['in_synapses'] = np.bincount(e['post'], weights=e['synapses'], minlength=c.n).astype(int)
    stats['pagerank'] = pagerank(c.matrix())
    return stats


def pagerank(W, alpha=0.85, tol=1e-10, max_iter=200):
    """Synapse-weighted PageRank. Dangling neurons redistribute uniformly."""
    n = W.shape[0]
    out = np.asarray(W.sum(axis=1)).ravel()
    dangling = out == 0
    P = sp.diags(np.where(dangling, 0, 1 / np.where(dangling, 1, out))) @ W
    PT = P.T.tocsr()
    r = np.full(n, 1 / n)
    for _ in range(max_iter):
        r_new = alpha * (PT @ r + r[dangling].sum() / n) + (1 - alpha) / n
        if np.abs(r_new - r).sum() < tol:
            return r_new
        r = r_new
    return r


def degree_ccdf(stats, column):
    """Complementary CDF P(X >= k) for a degree column, for log-log plots."""
    k, counts = np.unique(stats[column], return_counts=True)
    ccdf = counts[::-1].cumsum()[::-1] / counts.sum()
    keep = k > 0
    return pd.DataFrame({'k': k[keep], 'ccdf': ccdf[keep]})


# ---------------------------------------------------------------------------
# Class-level structure
# ---------------------------------------------------------------------------

def class_flow(c, column='super_class'):
    """Synapses between groups of neurons (rows = pre, columns = post), plus
    the fraction of those synapses that are inhibitory."""
    groups = c.neurons[column]
    e = c.edges
    df = pd.DataFrame({
        'pre': groups.to_numpy()[e['pre']],
        'post': groups.to_numpy()[e['post']],
        'synapses': e['synapses'].to_numpy(),
        'inhibitory': np.where(e['sign'] < 0, e['synapses'], 0),
    })
    g = df.groupby(['pre', 'post'], observed=True)[['synapses', 'inhibitory']].sum()
    g['inhibitory_frac'] = g['inhibitory'] / g['synapses']
    return g.reset_index()


def transmitter_composition(c):
    """Count of neurons per (super_class, predicted neurotransmitter)."""
    return (c.neurons.groupby(['super_class', 'top_nt'], observed=True)
            .size().rename('neurons').reset_index())


# ---------------------------------------------------------------------------
# Sensory-to-motor depth
# ---------------------------------------------------------------------------

def hops_from(W, sources):
    """Fewest-hop distance from any neuron in ``sources`` to every neuron.

    Implemented as single-source BFS from a virtual node wired to all sources.
    Unreachable neurons get ``inf``.
    """
    n = W.shape[0]
    B = (W > 0).astype(np.int8)
    seed = sp.csr_matrix((np.ones(len(sources), dtype=np.int8),
                          (np.zeros(len(sources), dtype=int), sources)), shape=(1, n))
    G = sp.vstack([B, seed])
    G = sp.hstack([G, sp.csr_matrix((n + 1, 1), dtype=np.int8)]).tocsr()
    d = shortest_path(G, unweighted=True, directed=True, indices=n)
    return d[:n] - 1


def sensory_depth(c, modalities=SENSORY_MODALITIES, min_synapses=STRONG):
    """For each sensory modality: how many hops it takes to reach each
    neuron class, as long-format rows (modality, target, hops, neurons)."""
    W = c.matrix(min_synapses)
    nt = c.neurons
    rows = []
    targets = {
        'whole brain': np.ones(c.n, bool),
        'descending': (nt['super_class'] == 'descending').to_numpy(),
        'motor': (nt['super_class'] == 'motor').to_numpy(),
    }
    for m in modalities:
        src = np.flatnonzero(((nt['super_class'] == 'sensory') & (nt['cell_class'] == m)).to_numpy())
        d = hops_from(W, src)
        for name, mask in targets.items():
            dm = d[mask]
            reached = np.isfinite(dm)
            counts = np.bincount(dm[reached].astype(int))
            for h, k in enumerate(counts):
                rows.append((m, len(src), name, int(mask.sum()), h, int(k)))
    return pd.DataFrame(rows, columns=['modality', 'sources', 'target', 'target_size',
                                       'hops', 'neurons'])


# ---------------------------------------------------------------------------
# Strongest shortest paths between two neuron sets
# ---------------------------------------------------------------------------

def strongest_paths(c, sources, targets, min_synapses=STRONG, k=25):
    """Top-k shortest paths from ``sources`` to ``targets`` (positions).

    Among all fewest-hop paths, ranks by bottleneck (the weakest connection
    on the path, in synapses), then by total synapses. Uses a layered k-best
    search restricted to neurons that lie on some shortest path.

    Returns (paths DataFrame, layer table of on-path neurons).
    """
    W = c.matrix(min_synapses)
    d_fwd = hops_from(W, sources)
    d_bwd = hops_from(W.T.tocsr(), targets)
    length = np.min(d_fwd[targets])
    if not np.isfinite(length):
        raise ValueError('targets are unreachable from sources')
    length = int(length)
    on_path = np.flatnonzero(np.isfinite(d_fwd) & np.isfinite(d_bwd) & (d_fwd + d_bwd == length))
    layer = {int(i): int(d_fwd[i]) for i in on_path}

    sub = W[on_path][:, on_path].tocoo()
    # best[node] = list of (bottleneck, total, path) sorted descending
    best = {int(on_path[i]): [] for i in range(len(on_path))}
    for i in on_path:
        if layer[int(i)] == 0:
            best[int(i)] = [(np.inf, 0.0, (int(i),))]
    edges_by_layer = {}
    for r, col, w in zip(sub.row, sub.col, sub.data):
        u, v = int(on_path[r]), int(on_path[col])
        if layer[v] == layer[u] + 1:
            edges_by_layer.setdefault(layer[u], []).append((u, v, float(w)))
    for h in range(length):
        incoming = {}
        for u, v, w in edges_by_layer.get(h, []):
            for b, t, p in best[u]:
                incoming.setdefault(v, []).append((min(b, w), t + w, p + (v,)))
        for v, cands in incoming.items():
            best[v] = sorted(cands, key=lambda x: (x[0], x[1]), reverse=True)[:k]

    finals = [x for t in targets if int(t) in best for x in best[int(t)]]
    finals = sorted(finals, key=lambda x: (x[0], x[1]), reverse=True)[:k]
    ids = c.neurons['root_id'].to_numpy()
    types = c.neurons['cell_type'].astype(str).to_numpy()
    paths = pd.DataFrame([{
        'rank': r + 1,
        'bottleneck_synapses': int(b),
        'total_synapses': int(t),
        'cell_types': ' > '.join(types[list(p)]),
        'root_ids': ' > '.join(str(ids[i]) for i in p),
    } for r, (b, t, p) in enumerate(finals)])

    layers = c.neurons.iloc[on_path][['root_id', 'cell_type', 'super_class', 'side']].copy()
    layers['layer'] = [layer[int(i)] for i in on_path]
    layers['index'] = on_path
    return paths, layers.sort_values(['layer', 'cell_type']).reset_index(drop=True)


def layer_type_flow(c, layers, min_synapses=STRONG):
    """Synapses between consecutive layers of a shortest-path subgraph,
    aggregated by cell type, for drawing a layered flow (Sankey) diagram."""
    idx = layers['index'].to_numpy()
    W = c.matrix(min_synapses)[idx][:, idx].tocoo()
    lay = layers['layer'].to_numpy()
    typ = layers['cell_type'].astype(str).to_numpy()
    df = pd.DataFrame({'src_layer': lay[W.row], 'src_type': typ[W.row],
                       'dst_layer': lay[W.col], 'dst_type': typ[W.col],
                       'synapses': W.data.astype(int)})
    df = df[df['dst_layer'] == df['src_layer'] + 1]
    return (df.groupby(['src_layer', 'src_type', 'dst_layer', 'dst_type'])['synapses']
            .sum().reset_index().sort_values('synapses', ascending=False))


# ---------------------------------------------------------------------------
# Left-right symmetry
# ---------------------------------------------------------------------------

def hemisphere_symmetry(c, min_synapses=1, min_total=50):
    """Compare left->left with right->right wiring at the cell-type level.

    For every (pre type, post type) pair of annotated cell types present in
    both hemispheres, returns ipsilateral synapse totals on each side.
    Every synapse counts (no per-connection threshold, which would zero out
    one side of many pairs); instead pairs with fewer than ``min_total``
    synapses over both sides are dropped as too sparse to compare.
    """
    nt = c.neurons
    # Work on integer codes: string comparisons over 15M edges are too slow.
    side = nt['side'].astype(str).to_numpy()
    side_code = np.select([side == 'left', side == 'right'], [0, 1], -1)
    types = nt['cell_type'].cat.categories
    typ = nt['cell_type'].cat.codes.to_numpy()
    bilateral = np.zeros(len(types), bool)
    bilateral[np.intersect1d(typ[side_code == 0], typ[side_code == 1])] = True
    if 'unknown' in types:
        bilateral[types.get_loc('unknown')] = False

    e = c.edges[c.edges['synapses'] >= min_synapses]
    pre, post = e['pre'].to_numpy(), e['post'].to_numpy()
    ps = side_code[pre]
    keep = (ps >= 0) & (ps == side_code[post]) & bilateral[typ[pre]] & bilateral[typ[post]]
    df = pd.DataFrame({'pre_type': types[typ[pre[keep]]],
                       'post_type': types[typ[post[keep]]],
                       'side': np.where(ps[keep] == 0, 'left', 'right'),
                       'synapses': e['synapses'].to_numpy()[keep]})
    wide = (df.pivot_table(index=['pre_type', 'post_type'], columns='side',
                           values='synapses', aggfunc='sum', fill_value=0)
            .reset_index())
    wide.columns.name = None
    return wide[wide['left'] + wide['right'] >= min_total].reset_index(drop=True)


def symmetry_score(wide):
    """Pearson r of log1p synapse counts, left vs right."""
    return float(np.corrcoef(np.log1p(wide['left']), np.log1p(wide['right']))[0, 1])
