import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _seq_aleatoria(rng, n):
    return "".join(rng.choice("ATCG") for _ in range(n))


@pytest.fixture
def df_genes():
    """300 genes sintéticos (sem 'N') com IDs no formato exportado pelo BioMart."""
    rng = random.Random(0)
    return pd.DataFrame({
        'Gene stable ID': [f"ENSG{i:011d}.1|GENE{i}" for i in range(300)],
        'Sequencia': [_seq_aleatoria(rng, 60) for _ in range(300)],
    })
