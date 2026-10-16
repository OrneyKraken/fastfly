"""Static figures (matplotlib) for the README and portfolio."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
import numpy as np  # noqa: E402

# Validated reference palette (light mode); see dashboard for dark mode.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
CONTEXT = '#d9d8d2'  # de-emphasised background marks
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
          '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BLUES = LinearSegmentedColormap.from_list('blues', [
    '#f0efec', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
    '#256abf', '#184f95', '#0d366b'])

SUPER_CLASS_ORDER = [
    'sensory', 'sensory_ascending', 'ascending', 'optic', 'visual_projection',
    'visual_centrifugal', 'central', 'descending', 'motor', 'endocrine']

LABELS = {
    'sensory_ascending': 'sensory ascending',
    'visual_projection': 'visual projection',
    'visual_centrifugal': 'visual centrifugal',
}


def label(s):
    return LABELS.get(s, s)


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'font.family': 'sans-serif',
        'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK_2,
        'axes.edgecolor': AXIS, 'axes.linewidth': 1,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 1,
        'axes.titlesize': 12, 'axes.titleweight': 'semibold',
        'axes.titlelocation': 'left', 'legend.frameon': False,
    })


def _save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def soma_map(xyz, super_class, path, classes=None):
    """Small multiples: each panel highlights one super-class against the
    whole brain, frontal view."""
    classes = classes or ['optic', 'central', 'sensory', 'visual_projection',
                          'ascending', 'descending', 'visual_centrifugal', 'motor']
    sc = np.asarray(super_class)
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.2))
    rng = np.random.default_rng(0)
    bg = rng.choice(len(xyz), size=min(40000, len(xyz)), replace=False)
    for ax, cls in zip(axes.ravel(), classes):
        ax.scatter(xyz[bg, 0], xyz[bg, 1], s=0.3, c=CONTEXT, lw=0, rasterized=True)
        m = np.flatnonzero(sc == cls)
        ax.scatter(xyz[m, 0], xyz[m, 1], s=0.6 if len(m) > 2000 else 4,
                   c=SERIES[0], lw=0, rasterized=True)
        ax.set_title(f'{label(cls)}  ·  {len(m):,}', fontsize=10)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
    fig.suptitle('Where each class of neuron lives (soma positions, frontal view)',
                 x=0.01, ha='left', fontsize=13, fontweight='semibold')
    _save(fig, path)


def degree_ccdf(ccdf_out, ccdf_in, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for df, name, col in [(ccdf_out, 'outgoing partners', SERIES[0]),
                          (ccdf_in, 'incoming partners', SERIES[1])]:
        ax.plot(df['k'], df['ccdf'], color=col, lw=2, label=name)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='major')
    ax.set_xlabel('number of partner neurons, k')
    ax.set_ylabel('fraction of neurons with ≥ k partners')
    ax.set_title('Connectivity is heavy-tailed: a few neurons talk to thousands')
    ax.legend(loc='lower left')
    _save(fig, path)


def class_flow(flow, path):
    order = [c for c in SUPER_CLASS_ORDER if c in set(flow['pre']) | set(flow['post'])]
    M = (flow.pivot(index='pre', columns='post', values='synapses')
         .reindex(index=order, columns=order).fillna(0).to_numpy())
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(np.where(M > 0, M, np.nan), cmap=BLUES,
                   norm=LogNorm(vmin=1e2, vmax=M.max()))
    ax.set_xticks(range(len(order)), [label(o) for o in order], rotation=40, ha='right')
    ax.set_yticks(range(len(order)), [label(o) for o in order])
    ax.set_xlabel('postsynaptic class (receives)')
    ax.set_ylabel('presynaptic class (sends)')
    ax.spines[:].set_visible(False)
    ax.tick_params(which='both', length=0, colors=INK_2)
    ax.set_xticks(np.arange(len(order) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(order) + 1) - .5, minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('synapses')
    cb.outline.set_visible(False)
    ax.set_title('Who talks to whom: synapses between neuron classes')
    _save(fig, path)


def transmitters(comp, path):
    nts = ['acetylcholine', 'glutamate', 'gaba', 'dopamine', 'serotonin', 'octopamine']
    order = [c for c in SUPER_CLASS_ORDER if c in set(comp['super_class'])]
    M = (comp.pivot(index='super_class', columns='top_nt', values='neurons')
         .reindex(index=order).fillna(0))
    M['other / unknown'] = M.drop(columns=[n for n in nts if n in M]).sum(axis=1)
    M = M[[n for n in nts if n in M] + ['other / unknown']]
    share = M.div(M.sum(axis=1), axis=0)
    colors = SERIES[:len(nts)] + [CONTEXT]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for col, colr in zip(share.columns, colors):
        ax.barh(y, share[col], left=left, height=0.62, color=colr,
                edgecolor=SURFACE, linewidth=1.5, label=col)
        left += share[col].to_numpy()
    ax.set_yticks(y, [f'{label(o)} ({int(M.loc[o].sum()):,})' for o in order])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis='y', length=0, colors=INK_2)
    ax.spines['left'].set_visible(False)
    ax.legend(ncol=4, loc='upper left', bbox_to_anchor=(0, -0.08), fontsize=9)
    ax.set_title('Predicted neurotransmitter by neuron class')
    _save(fig, path)


def sensory_depth(depth, path, target='descending'):
    d = depth[depth['target'] == target]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (m, g) in enumerate(d.groupby('modality', sort=False)):
        g = g.sort_values('hops')
        frac = g['neurons'].cumsum() / g['target_size'].iloc[0]
        ax.plot(g['hops'], frac, color=SERIES[i], lw=2, marker='o', ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=m)
    ax.set_xlabel('synaptic hops from sensory neurons')
    ax.set_ylabel(f'share of {target} neurons reached')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(-0.2, 6.5)
    ax.set_ylim(0, 1.02)
    ax.grid(True, axis='y')
    ax.legend(loc='lower right')
    ax.set_title(f'How fast each sense reaches the {target} neurons')
    _save(fig, path)


def symmetry(wide, r, path):
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    x, y = np.log10(wide['left'] + 1), np.log10(wide['right'] + 1)
    hb = ax.hexbin(x, y, gridsize=60, cmap=BLUES, bins='log', mincnt=1, linewidths=0)
    lim = max(x.max(), y.max())
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1)
    ticks = range(0, int(lim) + 1)
    ax.set_xticks(ticks, [f'{10**t:,}' for t in ticks])
    ax.set_yticks(ticks, [f'{10**t:,}' for t in ticks])
    ax.set_xlabel('synapses, left → left (+1)')
    ax.set_ylabel('synapses, right → right (+1)')
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label('cell-type pairs')
    cb.outline.set_visible(False)
    ax.set_title(f'Left and right brains are wired alike (r = {r:.2f})')
    ax.text(0, -0.2, f'{len(wide):,} cell-type pairs with ≥ 50 synapses; each hexagon counts pairs',
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    _save(fig, path)


def path_graph(layers, flow, path, title):
    """Layered drawing of the neurons on the shortest paths."""
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    n_layers = layers['layer'].max() + 1
    cls = dict(zip(layers['cell_type'].astype(str), layers['super_class'].astype(str)))
    pos = {}
    for h in range(n_layers):
        types = (layers[layers['layer'] == h]['cell_type'].astype(str)
                 .value_counts().sort_index())
        for j, (t, k) in enumerate(types.items()):
            pos[(h, t)] = (h, (len(types) - 1) / 2 - j, k)
    wmax = flow['synapses'].max()
    for _, r in flow.iterrows():
        x0, y0, _ = pos[(r['src_layer'], r['src_type'])]
        x1, y1, _ = pos[(r['dst_layer'], r['dst_type'])]
        ax.plot([x0, x1], [y0, y1], color=SERIES[0], alpha=0.55,
                lw=0.6 + 5 * r['synapses'] / wmax, solid_capstyle='round', zorder=1)
    for (h, t), (x, y, k) in pos.items():
        ax.scatter(x, y, s=120 + 40 * k, color=SURFACE, edgecolor=INK_2, lw=1.2, zorder=2)
        ax.text(x, y - 0.16, f'{t}' + (f' ×{k}' if k > 1 else ''), ha='center',
                va='top', fontsize=9.5, color=INK)
        ax.text(x, y - 0.33, label(cls[t]), ha='center', va='top', fontsize=8, color=MUTED)
    ax.xaxis.tick_top()
    ax.set_xticks(range(n_layers), ['sugar receptor neurons'] +
                  [f'interneurons (hop {h})' for h in range(1, n_layers - 1)] +
                  ['MN9 feeding motor neuron'])
    ax.tick_params(axis='x', length=0, colors=INK_2, pad=8)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.3)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_title(title, pad=30)
    ax.text(0, -0.04, 'Line width = synapses (connections ≥ 5 synapses). '
            'Node size = number of neurons of that type.',
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    _save(fig, path)
