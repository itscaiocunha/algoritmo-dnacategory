"""Imputação de bases desconhecidas ('N') por K-Means + KNN ponderado.

1. Sequências sem 'N' formam a referência. Elas são vetorizadas em k-mers e
   agrupadas por K-Means em "famílias" de genes parecidos.
2. Para cada sequência com 'N', prevemos seu cluster, buscamos os k vizinhos
   mais próximos (distância de Hamming) dentro dele e escolhemos cada base
   faltante por voto ponderado pelo inverso da distância.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from .features import get_kmers_string, sequencias_para_kmers
from .ids import limpar_id

BASES = ('A', 'T', 'C', 'G')


def distancia_hamming(s1, s2):
    """Número de posições diferentes entre duas sequências.

    Para comprimentos diferentes, usa uma penalidade aproximada: diferença de
    tamanho + metade do maior comprimento.
    """
    if len(s1) != len(s2):
        return abs(len(s1) - len(s2)) + max(len(s1), len(s2)) // 2
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def voto_majoritario_ponderado(vizinhos_seqs, vizinhos_dist, pos_i):
    """Base mais votada na posição `pos_i`, com peso 1 / (distância + 1).

    Retorna 'N' se nenhum vizinho tiver uma base válida nessa posição.
    """
    votos = dict.fromkeys(BASES, 0.0)
    for seq, dist in zip(vizinhos_seqs, vizinhos_dist):
        if len(seq) > pos_i and seq[pos_i] in votos:
            votos[seq[pos_i]] += 1.0 / (dist + 1.0)

    votos_validos = {base: peso for base, peso in votos.items() if peso > 0}
    if not votos_validos:
        return 'N'
    return max(votos_validos, key=votos_validos.get)


# --- Treino ---

def treinar_imputador(df_genes, kmer_size, n_clusters, random_state=42, min_sequencias=1000):
    """Treina o vetorizador de k-mers e o K-Means nas sequências sem 'N'.

    Retorna (vectorizer, kmeans, df_referencia), onde df_referencia tem as
    colunas ['join_key', 'Sequencia', 'Cluster'].
    """
    df = df_genes.copy()
    df['Sequencia'] = df['Sequencia'].astype(str).fillna('')
    df_limpo = df[~df['Sequencia'].str.contains('N')].copy()

    if len(df_limpo) < min_sequencias:
        raise ValueError(
            f"Dados limpos insuficientes para treinar ({len(df_limpo)} < {min_sequencias})."
        )
    print(f"Total de {len(df_limpo)} sequências limpas para treinamento.")

    print(f"Vetorizando k-mers (k={kmer_size})...")
    vectorizer = CountVectorizer(analyzer='word')
    kmer_vectors = vectorizer.fit_transform(sequencias_para_kmers(df_limpo['Sequencia'], kmer_size))
    print(f"Matriz de features: {kmer_vectors.shape}")

    print(f"Treinando K-Means (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_limpo['Cluster'] = kmeans.fit_predict(kmer_vectors)

    df_limpo['join_key'] = df_limpo.iloc[:, 0].apply(limpar_id)
    df_referencia = df_limpo[['join_key', 'Sequencia', 'Cluster']].dropna(subset=['join_key'])
    return vectorizer, kmeans, df_referencia


def salvar_imputador(vectorizer, kmeans, df_referencia, path_vectorizer, path_kmeans, path_referencia):
    Path(path_vectorizer).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path_vectorizer)
    joblib.dump(kmeans, path_kmeans)
    df_referencia.to_csv(path_referencia, index=False)
    print(f"Imputador salvo em: {path_vectorizer}, {path_kmeans}, {path_referencia}")


def carregar_imputador(path_vectorizer, path_kmeans, path_referencia):
    return (
        joblib.load(path_vectorizer),
        joblib.load(path_kmeans),
        pd.read_csv(path_referencia),
    )


# --- Aplicação ---

def imputar_sequencia(sequencia, vectorizer, kmeans, df_referencia, kmer_size, k_vizinhos,
                      random_state=42):
    """Preenche os 'N' de uma sequência.

    Retorna (nova_sequencia, cluster_id, n_imputados). Bases sem voto válido
    permanecem como 'N'.
    """
    indices_n = [i for i, base in enumerate(sequencia) if base == 'N']
    if not indices_n:
        return sequencia, None, 0

    vetor_kmer = vectorizer.transform([get_kmers_string(sequencia, kmer_size)])
    cluster_id = kmeans.predict(vetor_kmer)[0]

    candidatos = df_referencia[df_referencia['Cluster'] == cluster_id]
    if len(candidatos) < k_vizinhos:
        # Cluster pequeno demais: busca numa amostra aleatória da referência toda
        candidatos = df_referencia.sample(n=k_vizinhos * 2, random_state=random_state)

    distancias = [distancia_hamming(sequencia, ref) for ref in candidatos['Sequencia']]
    indices_vizinhos = np.argsort(distancias)[:k_vizinhos]
    seqs_vizinhas = candidatos.iloc[indices_vizinhos]['Sequencia'].tolist()
    dists_vizinhas = [distancias[i] for i in indices_vizinhos]

    nova_sequencia = list(sequencia)
    n_imputados = 0
    for i in indices_n:
        base = voto_majoritario_ponderado(seqs_vizinhas, dists_vizinhas, i)
        if base != 'N':
            nova_sequencia[i] = base
            n_imputados += 1
    return "".join(nova_sequencia), cluster_id, n_imputados


def _imputar_linha(linha, vectorizer, kmeans, df_referencia, kmer_size, k_vizinhos):
    sequencia = linha['Sequencia']
    if 'N' not in str(sequencia):
        return linha
    try:
        nova, cluster_id, n_imputados = imputar_sequencia(
            sequencia, vectorizer, kmeans, df_referencia, kmer_size, k_vizinhos
        )
        linha['Sequencia'] = nova
        print(f"  -> ID {linha['join_key']} (Cluster {cluster_id}): {n_imputados} bases 'N' imputadas.")
    except Exception as e:
        print(f"  -> ERRO processando {linha['join_key']}: {e}. Pulando linha.")
    return linha


def _carregar_progresso(path_progresso):
    """Índice do próximo chunk a processar (0 se não houver checkpoint)."""
    try:
        return int(Path(path_progresso).read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def aplicar_imputador(path_entrada, path_saida, path_progresso, vectorizer, kmeans, df_referencia,
                      kmer_size, k_vizinhos, chunk_size):
    """Imputa os 'N' de um CSV grande, chunk a chunk, com checkpoint.

    Se o processo for interrompido, a próxima execução retoma a partir do
    último chunk salvo (registrado em `path_progresso`).
    """
    path_saida = Path(path_saida)
    path_progresso = Path(path_progresso)

    chunks_completos = _carregar_progresso(path_progresso)
    if chunks_completos == 0:
        print("Nenhum checkpoint encontrado. Começando do zero.")
        path_saida.unlink(missing_ok=True)
        modo_escrita, escrever_header = 'w', True
    else:
        print(f"Checkpoint encontrado. Retomando a partir do chunk {chunks_completos}.")
        modo_escrita, escrever_header = 'a', False

    reader = pd.read_csv(path_entrada, chunksize=chunk_size)
    for i, chunk in enumerate(reader):
        if i < chunks_completos:
            print(f"Pulando chunk {i} (já processado)")
            continue

        print(f"\n--- Chunk {i} (linhas {i * chunk_size} a {(i + 1) * chunk_size}) ---")
        if 'join_key' not in chunk.columns:
            chunk['join_key'] = chunk.iloc[:, 0].apply(limpar_id)

        chunk_limpo = chunk.apply(
            _imputar_linha, axis=1,
            args=(vectorizer, kmeans, df_referencia, kmer_size, k_vizinhos),
        )
        chunk_limpo.to_csv(path_saida, mode=modo_escrita, index=False, header=escrever_header)
        path_progresso.write_text(str(i + 1))
        modo_escrita, escrever_header = 'a', False

    # Só remove o checkpoint se tudo terminou; em caso de erro, ele permite retomar
    path_progresso.unlink(missing_ok=True)
    print(f"\nArquivo imputado salvo em: {path_saida}")
