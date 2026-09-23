import pandas as pd

from dnacategory.analise import calcular_conteudo_gc, relatorio
from dnacategory.classificacao import balancear_undersampling, treinar_svm
from dnacategory.dataset import achatar_dataset, juntar_dados


def test_juntar_e_achatar(tmp_path, df_genes):
    genes = df_genes.head(4)
    ids = [f"ENSG{i:011d}" for i in range(4)]
    path_seq = tmp_path / "genes.csv"
    path_funcao = tmp_path / "funcao.csv"
    path_orto = tmp_path / "orto.csv"
    genes.to_csv(path_seq, index=False)
    pd.DataFrame({
        'Gene stable ID': [ids[0], ids[0], ids[1], ids[2]],
        'GO term name': ['protein binding', 'nucleus', 'nucleus', 'membrane'],
        'GO domain': ['molecular_function', 'cellular_component', 'cellular_component', 'cellular_component'],
    }).to_csv(path_funcao, index=False)
    pd.DataFrame({'Gene stable ID': ids, 'Mouse gene stable ID': ['M1', 'M2', 'M3', 'M4']}) \
        .to_csv(path_orto, index=False)

    df_mestre = juntar_dados(path_seq, path_funcao, path_orto)
    # gene 3 não tem GO term e sai; gene 0 aparece duas vezes (duas anotações)
    assert len(df_mestre) == 4
    assert set(df_mestre['join_key']) == set(ids[:3])

    path_mestre = tmp_path / "mestre.csv"
    df_mestre.to_csv(path_mestre, index=False)
    df_flat = achatar_dataset(path_mestre, 'protein binding', chunk_size=2)

    assert df_flat['label'].to_dict() == {ids[0]: 1, ids[1]: 0, ids[2]: 0}


def test_balancear_undersampling():
    df = pd.DataFrame({'Sequencia': ['A'] * 10, 'label': [1] * 8 + [0] * 2})
    assert balancear_undersampling(df)['label'].value_counts().to_dict() == {0: 2, 1: 2}


def test_treinar_svm_sinal_separavel(df_genes):
    # Classe 1 tem um motivo repetido: o SVM deve separar quase perfeitamente
    df = df_genes[['Sequencia']].copy()
    df['label'] = [i % 2 for i in range(len(df))]
    df.loc[df['label'] == 1, 'Sequencia'] = "GGGGGGGGGG" + df['Sequencia']

    resultado = treinar_svm(df, kmer_size=3)
    assert resultado.acuracia > 0.9
    assert len(resultado.y_scores) == len(resultado.y_test)


def test_conteudo_gc():
    assert calcular_conteudo_gc("GGCC") == 100.0
    assert calcular_conteudo_gc("ATGC") == 50.0
    assert calcular_conteudo_gc("GCNN") == 100.0  # N fora do denominador
    assert calcular_conteudo_gc("") == 0.0
    assert calcular_conteudo_gc(None) == 0.0


def test_relatorio_trata_sequencia_vazia():
    df = pd.DataFrame({'id': ['a', 'b'], 'Sequencia': ['ATGN', None]})
    resultado = relatorio(df)
    assert resultado['Comprimento'].tolist() == [4, 0]
    assert resultado['N_Percent_%'].tolist() == [25.0, 0.0]
