"""Tests on a tiny hand-built connectome, so they run without the real data."""

import numpy as np
import pandas as pd
import pytest

from flyconn import analysis as A
from flyconn.data import Connectome


def make(edges, n, side=None, cell_type=None, super_class=None):
    neurons = pd.DataFrame({
        'root_id': np.arange(100, 100 + n),
        'super_class': pd.Categorical(super_class or ['central'] * n),
        'cell_class': pd.Categorical(['x'] * n),
        'cell_type': pd.Categorical(cell_type or [f't{i}' for i in range(n)]),
        'top_nt': pd.Categorical(['acetylcholine'] * n),
        'side': pd.Categorical(side or ['left'] * n),
    })
    pre, post, syn, sign = zip(*edges)
    e = pd.DataFrame({'pre': np.array(pre, np.int32), 'post': np.array(post, np.int32),
                      'synapses': np.array(syn, np.int32), 'sign': np.array(sign, np.int8)})
    return Connectome(neurons=neurons, edges=e)


def test_matrix_threshold_and_sign():
    c = make([(0, 1, 10, 1), (1, 2, 3, -1)], 3)
    assert c.matrix().nnz == 2
    assert c.matrix(min_synapses=5).nnz == 1
    assert c.matrix(signed=True)[1, 2] == -3


def test_index_of():
    c = make([(0, 1, 1, 1)], 3)
    assert list(c.index_of([102, 100])) == [2, 0]
    with pytest.raises(KeyError):
        c.index_of([999])


def test_pagerank_sums_to_one_and_ranks_sink():
    # star: everyone feeds neuron 0
    c = make([(i, 0, 5, 1) for i in range(1, 6)], 6)
    r = A.pagerank(c.matrix())
    assert r.sum() == pytest.approx(1.0)
    assert r.argmax() == 0


def test_hops_from_multiple_sources():
    # 0 -> 1 -> 2 -> 3, and 4 -> 3 ; 5 unreachable
    c = make([(0, 1, 5, 1), (1, 2, 5, 1), (2, 3, 5, 1), (4, 3, 5, 1)], 6)
    d = A.hops_from(c.matrix(), np.array([0, 4]))
    assert list(d[:5]) == [0, 1, 2, 1, 0]
    assert np.isinf(d[5])


def test_strongest_paths_prefers_bigger_bottleneck():
    # two 2-hop routes 0->3: via 1 (bottleneck 6) and via 2 (bottleneck 9)
    c = make([(0, 1, 50, 1), (1, 3, 6, 1), (0, 2, 9, 1), (2, 3, 20, 1)], 4)
    paths, layers = A.strongest_paths(c, np.array([0]), np.array([3]))
    assert paths.iloc[0]['root_ids'] == '100 > 102 > 103'
    assert paths.iloc[0]['bottleneck_synapses'] == 9
    assert list(paths['bottleneck_synapses']) == [9, 6]
    assert set(layers['layer']) == {0, 1, 2}


def test_strongest_paths_ignores_weak_edges():
    c = make([(0, 1, 2, 1), (1, 2, 50, 1)], 3)
    with pytest.raises(ValueError):
        A.strongest_paths(c, np.array([0]), np.array([2]))


def test_hemisphere_symmetry_counts_ipsilateral_only():
    # types a, b exist on both sides; one contralateral edge must be ignored
    side = ['left', 'left', 'right', 'right']
    types = ['a', 'b', 'a', 'b']
    c = make([(0, 1, 40, 1), (2, 3, 30, 1), (0, 3, 99, 1)], 4, side=side, cell_type=types)
    wide = A.hemisphere_symmetry(c, min_total=0)
    row = wide[(wide['pre_type'] == 'a') & (wide['post_type'] == 'b')].iloc[0]
    assert (row['left'], row['right']) == (40, 30)
    assert len(wide) == 1


def test_class_flow_inhibitory_fraction():
    sc = ['sensory', 'central', 'central']
    c = make([(0, 1, 10, 1), (0, 2, 30, -1)], 3, super_class=sc)
    f = A.class_flow(c).set_index(['pre', 'post'])
    assert f.loc[('sensory', 'central'), 'synapses'] == 40
    assert f.loc[('sensory', 'central'), 'inhibitory_frac'] == pytest.approx(0.75)
