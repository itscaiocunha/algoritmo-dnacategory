import pandas as pd
import pytest

from dnacategory.features import get_kmers_string, sequencias_para_kmers
from dnacategory.ids import limpar_id


@pytest.mark.parametrize("entrada, esperado", [
    ("ENSG00000139618", "ENSG00000139618"),
    ("ENSG00000139618.15", "ENSG00000139618"),
    ("ENSG00000139618.15|BRCA2|chr13", "ENSG00000139618"),
    ("ENSG00000139618 descricao", "ENSG00000139618"),
    ("ENST00000380152", None),
    ("", None),
    ("   ", None),
    (float("nan"), None),
])
def test_limpar_id(entrada, esperado):
    assert limpar_id(entrada) == esperado


def test_get_kmers_string():
    assert get_kmers_string("ATGCA", 3) == "ATG TGC GCA"


def test_get_kmers_string_sequencia_menor_que_k():
    assert get_kmers_string("AT", 3) == ""


def test_sequencias_para_kmers():
    resultado = sequencias_para_kmers(pd.Series(["AAAA", "ATGC"]), 2)
    assert resultado.tolist() == ["AA AA AA", "AT TG GC"]
