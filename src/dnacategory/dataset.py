"""Montagem do dataset: merge das fontes do BioMart e achatamento por gene."""
import numpy as np
import pandas as pd

from .ids import limpar_id


def _carregar_com_join_key(caminho, coluna_id=None):
    """Lê um CSV e cria 'join_key' a partir de `coluna_id` (padrão: 1ª coluna)."""
    df = pd.read_csv(caminho)
    coluna_id = coluna_id or df.columns[0]
    df['join_key'] = df[coluna_id].apply(limpar_id)
    return df.dropna(subset=['join_key'])


def juntar_dados(path_sequencias, path_funcoes, path_ortologos):
    """Junta sequências, anotações GO e ortólogos em um dataset "longo".

    O resultado tem uma linha por par (gene, GO term): a sequência de um
    gene se repete para cada anotação. Genes sem nenhum GO term são removidos.
    """
    print(f"Carregando sequências de {path_sequencias}...")
    df_seq = _carregar_com_join_key(path_sequencias)[['join_key', 'Sequencia']]
    print(f"Encontradas {len(df_seq)} sequências com IDs limpos.")

    print(f"Carregando funções de {path_funcoes}...")
    df_funcao = _carregar_com_join_key(path_funcoes, 'Gene stable ID')[['join_key', 'GO term name', 'GO domain']]
    print(f"Encontradas {len(df_funcao)} anotações de função (GO terms).")

    print(f"Carregando ortólogos de {path_ortologos}...")
    df_ortologos = _carregar_com_join_key(path_ortologos, 'Gene stable ID').drop(columns=['Gene stable ID'])
    print(f"Encontradas {len(df_ortologos)} anotações de ortólogos.")

    print("Iniciando merge dos datasets...")
    df_mestre = pd.merge(df_seq, df_funcao, on='join_key', how='left')
    df_ortologos_unicos = df_ortologos.drop_duplicates(subset=['join_key'])
    df_mestre = pd.merge(df_mestre, df_ortologos_unicos, on='join_key', how='left')
    df_mestre = df_mestre.dropna(subset=['GO term name'])

    print(f"Total de linhas (gene + anotação): {len(df_mestre)}")
    print(f"Total de genes únicos: {df_mestre['join_key'].nunique()}")
    return df_mestre


def top_funcoes_moleculares(df_mestre, n=10):
    """Contagem dos GO terms mais frequentes no domínio 'molecular_function'."""
    mf = df_mestre[df_mestre['GO domain'] == 'molecular_function']
    return mf['GO term name'].value_counts().head(n)


def achatar_dataset(path_entrada, target_function, chunk_size):
    """Reduz o dataset longo para uma linha por gene com um rótulo binário.

    label = 1 se *qualquer* anotação do gene for `target_function`, senão 0.
    O arquivo é lido em chunks para caber na memória (~14M linhas).
    """
    colunas = ['join_key', 'Sequencia', 'GO term name']
    labels_parciais = []
    seqs_parciais = []

    reader = pd.read_csv(path_entrada, usecols=colunas, chunksize=chunk_size, low_memory=False)
    for i, chunk in enumerate(reader):
        print(f"Processando chunk {i}...")
        chunk['GO term name'] = chunk['GO term name'].fillna('Sem_Funcao')
        chunk['label'] = np.where(chunk['GO term name'] == target_function, 1, 0)
        labels_parciais.append(chunk.groupby('join_key')['label'].max())
        seqs_parciais.append(chunk.groupby('join_key')['Sequencia'].first())

    print("Consolidando chunks...")
    # Um gene pode aparecer em mais de um chunk: agrupa de novo
    labels = pd.concat(labels_parciais).groupby(level=0).max()
    seqs = pd.concat(seqs_parciais).groupby(level=0).first()
    df_final = pd.merge(seqs, labels, left_index=True, right_index=True, how='inner')

    print(f"Dataset achatado para {len(df_final)} genes únicos.")
    print(f"Distribuição das classes:\n{df_final['label'].value_counts()}\n")
    return df_final
