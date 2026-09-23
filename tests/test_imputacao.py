import pandas as pd
import pytest

from dnacategory.imputacao import (
    aplicar_imputador,
    distancia_hamming,
    imputar_sequencia,
    treinar_imputador,
    voto_majoritario_ponderado,
)


def test_hamming_mesmo_tamanho():
    assert distancia_hamming("ATGC", "ATGC") == 0
    assert distancia_hamming("ATGC", "TTGA") == 2


def test_hamming_tamanhos_diferentes():
    # |4 - 6| + 6 // 2
    assert distancia_hamming("ATGC", "ATGCAA") == 5


def test_voto_ponderado_vizinho_mais_proximo_pesa_mais():
    # 'A' vem de um vizinho a distância 0 (peso 1); 'C' de dois a distância 5 (peso 1/6 cada)
    assert voto_majoritario_ponderado(["A", "C", "C"], [0, 5, 5], 0) == "A"


def test_voto_ponderado_ignora_n_e_posicao_fora():
    assert voto_majoritario_ponderado(["N", "AT"], [0, 0], 1) == "T"
    assert voto_majoritario_ponderado(["N", "A"], [0, 0], 3) == "N"


@pytest.fixture
def imputador(df_genes):
    return treinar_imputador(df_genes, kmer_size=3, n_clusters=3, min_sequencias=100)


def test_treinar_imputador(df_genes, imputador):
    _, _, df_ref = imputador
    assert list(df_ref.columns) == ['join_key', 'Sequencia', 'Cluster']
    assert len(df_ref) == len(df_genes)
    assert df_ref['join_key'].str.match(r"^ENSG\d+$").all()


def test_treinar_imputador_poucos_dados(df_genes):
    with pytest.raises(ValueError):
        treinar_imputador(df_genes, kmer_size=3, n_clusters=3, min_sequencias=1000)


def test_imputar_sequencia_recupera_base(df_genes, imputador):
    vectorizer, kmeans, df_ref = imputador
    original = df_genes['Sequencia'].iloc[0]
    furada = "N" + original[1:]

    nova, _, n_imputados = imputar_sequencia(furada, vectorizer, kmeans, df_ref, 3, 5)
    # A própria sequência original está na referência, a distância 1: ela vence o voto
    assert n_imputados == 1
    assert nova == original


def test_imputar_sequencia_sem_n_nao_muda(imputador):
    vectorizer, kmeans, df_ref = imputador
    assert imputar_sequencia("ATGC", vectorizer, kmeans, df_ref, 3, 5) == ("ATGC", None, 0)


def test_aplicar_imputador_com_checkpoint(tmp_path, df_genes, imputador):
    vectorizer, kmeans, df_ref = imputador
    entrada = tmp_path / "mestre.csv"
    saida = tmp_path / "imputado.csv"
    progresso = tmp_path / "progresso.txt"

    df = df_genes.head(10).copy()
    df['join_key'] = [f"ENSG{i:011d}" for i in range(10)]
    df.loc[3, 'Sequencia'] = "N" + df.loc[3, 'Sequencia'][1:]
    df.to_csv(entrada, index=False)

    # Simula uma execução interrompida depois do 1º chunk (linhas 0-3)
    progresso.write_text("1")
    df.head(4).to_csv(saida, index=False)

    aplicar_imputador(entrada, saida, progresso, vectorizer, kmeans, df_ref,
                      kmer_size=3, k_vizinhos=5, chunk_size=4)

    resultado = pd.read_csv(saida)
    assert len(resultado) == 10
    assert resultado['Sequencia'].iloc[4:].tolist() == df['Sequencia'].iloc[4:].tolist()
    assert not progresso.exists()
