"""Load the FlyWire v783 connectome and neuron annotations into one object."""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw'
CONNECTIVITY = DATA_DIR / 'flywire_783' / '2025_Connectivity_783.parquet'
NEURONS = DATA_DIR / 'flywire_783' / '2025_Completeness_783.csv'
ANNOTATIONS = DATA_DIR / 'annotations' / 'neuron_annotations.tsv'

# FlyWire coordinates are in 4 x 4 x 40 nm voxels.
VOXEL_NM = np.array([4.0, 4.0, 40.0])

# Columns kept from the annotation table.
ANNOTATION_COLUMNS = [
    'flow', 'super_class', 'cell_class', 'cell_type', 'top_nt', 'side',
    'nerve', 'soma_x', 'soma_y', 'soma_z', 'pos_x', 'pos_y', 'pos_z',
]

# Sugar-sensing gustatory receptor neurons used in the fly-brain "sugar"
# experiment (Shiu et al. 2024).
SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940639259967, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]

# MN9, the proboscis motor neuron driven by sugar (FlyWire type CB0701).
MN9 = [720575940660219265, 720575940618238523]


@dataclass
class Connectome:
    """Neuron table plus the edge list of neuron-to-neuron connections.

    ``neurons`` is indexed by position (0..N-1), which is the index used in
    ``edges`` (``pre``/``post``) and in every sparse matrix.
    """

    neurons: pd.DataFrame
    edges: pd.DataFrame

    @property
    def n(self):
        return len(self.neurons)

    def index_of(self, root_ids):
        """Positions of FlyWire root IDs (raises if any is unknown)."""
        lookup = pd.Series(np.arange(self.n), index=self.neurons['root_id'])
        return lookup.loc[list(root_ids)].to_numpy()

    def matrix(self, min_synapses=1, signed=False):
        """Sparse CSR matrix W[pre, post] of synapse counts.

        Connections with fewer than ``min_synapses`` synapses are dropped.
        With ``signed=True`` inhibitory connections are negative.
        """
        e = self.edges[self.edges['synapses'] >= min_synapses]
        w = e['synapses'].to_numpy(dtype=np.float32)
        if signed:
            w = w * e['sign'].to_numpy()
        return sp.csr_matrix((w, (e['pre'], e['post'])), shape=(self.n, self.n))

    @cached_property
    def soma_um(self):
        """(N, 3) soma positions in micrometres; falls back to a point on the
        neuron where the soma position is missing."""
        soma = self.neurons[['soma_x', 'soma_y', 'soma_z']].to_numpy(float)
        pos = self.neurons[['pos_x', 'pos_y', 'pos_z']].to_numpy(float)
        xyz = np.where(np.isnan(soma), pos, soma)
        return xyz * VOXEL_NM / 1000.0


def load(connectivity=CONNECTIVITY, neurons=NEURONS, annotations=ANNOTATIONS):
    """Load the connectome. Takes a few seconds and ~2 GB of RAM."""
    ids = pd.read_csv(neurons, index_col=0).index.to_numpy()
    ann = (pd.read_csv(annotations, sep='\t', low_memory=False)
           .drop_duplicates('root_id')
           .set_index('root_id')[ANNOTATION_COLUMNS])
    table = ann.reindex(ids).rename_axis('root_id').reset_index()
    for col in ['flow', 'super_class', 'cell_class', 'cell_type', 'top_nt', 'side']:
        table[col] = table[col].fillna('unknown').astype('category')

    raw = pd.read_parquet(connectivity, columns=[
        'Presynaptic_Index', 'Postsynaptic_Index', 'Connectivity', 'Excitatory'])
    edges = pd.DataFrame({
        'pre': raw['Presynaptic_Index'].to_numpy(np.int32),
        'post': raw['Postsynaptic_Index'].to_numpy(np.int32),
        'synapses': raw['Connectivity'].to_numpy(np.int32),
        'sign': raw['Excitatory'].to_numpy(np.int8),
    })
    return Connectome(neurons=table, edges=edges)
