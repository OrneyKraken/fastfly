"""Run every analysis and write figures, tables and the dashboard data file.

    python run_analysis.py            # ~2 minutes, ~4 GB RAM
"""

import json
import time
from pathlib import Path

import numpy as np

import flyconn
from flyconn import analysis as A
from flyconn import plots as P

OUT = Path(__file__).resolve().parent / 'outputs'
FIG = OUT / 'figures'
TAB = OUT / 'tables'


def step(msg, t0=[time.time()]):
    print(f'[{time.time() - t0[0]:6.1f}s] {msg}', flush=True)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    P.style()

    step('loading connectome')
    c = flyconn.load()

    step('summary statistics')
    summary = A.summary(c)

    step('per-neuron stats + PageRank')
    stats = A.neuron_stats(c)
    hubs = stats.sort_values('pagerank', ascending=False).head(50)
    hubs.to_csv(TAB / 'top_neurons_pagerank.csv', index=False)
    ccdf_out = A.degree_ccdf(stats, 'out_partners')
    ccdf_in = A.degree_ccdf(stats, 'in_partners')
    P.degree_ccdf(ccdf_out, ccdf_in, FIG / 'degree_distribution.png')

    step('soma map')
    P.soma_map(c.soma_um, c.neurons['super_class'].astype(str), FIG / 'soma_map.png')

    step('class-to-class flow')
    flow = A.class_flow(c)
    flow.to_csv(TAB / 'super_class_flow.csv', index=False)
    P.class_flow(flow, FIG / 'class_flow.png')

    step('neurotransmitters')
    comp = A.transmitter_composition(c)
    comp.to_csv(TAB / 'transmitters_by_class.csv', index=False)
    P.transmitters(comp, FIG / 'transmitters.png')

    step('sensory depth (BFS)')
    depth = A.sensory_depth(c)
    depth.to_csv(TAB / 'sensory_depth.csv', index=False)
    P.sensory_depth(depth, FIG / 'sensory_depth.png')

    step('sugar -> MN9 strongest paths')
    paths, layers = A.strongest_paths(c, c.index_of(flyconn.SUGAR_GRNS), c.index_of(flyconn.MN9))
    layer_flow = A.layer_type_flow(c, layers)
    paths.to_csv(TAB / 'sugar_to_mn9_paths.csv', index=False)
    layers.drop(columns='index').to_csv(TAB / 'sugar_to_mn9_neurons.csv', index=False)
    P.path_graph(layers, layer_flow, FIG / 'sugar_to_mn9.png',
                 'From sugar taste to the feeding motor neuron in two synapses')

    step('hemisphere symmetry')
    wide = A.hemisphere_symmetry(c)
    r = A.symmetry_score(wide)
    summary['symmetry_r'] = r
    summary['symmetry_pairs'] = int(len(wide))
    P.symmetry(wide, r, FIG / 'symmetry.png')

    step('writing dashboard data')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    data = dashboard_data(c, summary, hubs, ccdf_out, ccdf_in, flow, comp,
                          depth, paths, layers, layer_flow, wide)
    (OUT / 'dashboard_data.json').write_text(json.dumps(data, separators=(',', ':')))
    step('done')


def thin(df, n=80):
    """Log-spaced subsample of a CCDF curve for the dashboard."""
    idx = np.unique(np.geomspace(1, len(df), n).astype(int) - 1)
    return df.iloc[idx]


def dashboard_data(c, summary, hubs, ccdf_out, ccdf_in, flow, comp, depth,
                   paths, layers, layer_flow, wide, n_soma=60000):
    rng = np.random.default_rng(0)
    xyz = c.soma_um
    sc = c.neurons['super_class'].astype(str).to_numpy()
    ok = np.flatnonzero(np.isfinite(xyz).all(axis=1))
    # Keep every neuron of the small classes; subsample the big ones.
    small = ok[~np.isin(sc[ok], ['optic', 'central', 'sensory'])]
    big = ok[np.isin(sc[ok], ['optic', 'central', 'sensory'])]
    pick = np.concatenate([small, rng.choice(big, n_soma - len(small), replace=False)])
    classes = [k for k in P.SUPER_CLASS_ORDER if k in set(sc)]
    code = {k: i for i, k in enumerate(classes)}
    types = c.neurons['cell_type'].cat.categories.astype(str).tolist()
    tcode = c.neurons['cell_type'].cat.codes.to_numpy()
    # [x µm, y µm, class index, cell-type index]
    soma = [[round(float(xyz[i, 0])), round(float(xyz[i, 1])), code[sc[i]], int(tcode[i])]
            for i in pick]

    # Hexbin-style 2D histogram of left vs right synapses (log10 bins).
    lx, ly = np.log10(wide['left'] + 1), np.log10(wide['right'] + 1)
    edges = np.linspace(0, float(max(lx.max(), ly.max())) + 1e-9, 41)
    H, _, _ = np.histogram2d(lx, ly, bins=[edges, edges])

    return {
        'summary': summary,
        'classes': classes,
        'class_counts': {k: int((sc == k).sum()) for k in classes},
        'soma': soma,
        'types': types,
        'flow': flow.assign(pre=flow['pre'].astype(str), post=flow['post'].astype(str))
                    .to_dict('records'),
        'ccdf': {'out': thin(ccdf_out).values.tolist(), 'in': thin(ccdf_in).values.tolist()},
        'transmitters': comp.assign(super_class=comp['super_class'].astype(str),
                                    top_nt=comp['top_nt'].astype(str)).to_dict('records'),
        'depth': depth.to_dict('records'),
        'hubs': hubs.assign(**{k: hubs[k].astype(str) for k in
                               ['root_id', 'super_class', 'cell_class', 'cell_type', 'top_nt', 'side']})
                    .to_dict('records'),
        'paths': paths.to_dict('records'),
        'path_neurons': layers.drop(columns='index').astype({'root_id': str, 'cell_type': str,
                                                             'super_class': str, 'side': str})
                              .to_dict('records'),
        'path_flow': layer_flow.to_dict('records'),
        'symmetry': {'edges': edges.round(3).tolist(), 'counts': H.astype(int).tolist(),
                     'r': summary['symmetry_r'], 'pairs': summary['symmetry_pairs']},
    }


if __name__ == '__main__':
    main()
