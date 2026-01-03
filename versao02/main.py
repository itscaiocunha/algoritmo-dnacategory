import pandas as pd
import numpy as np
from collections import Counter
import sys
import random
import os

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    print("ERRO: A biblioteca 'scikit-learn' não foi encontrada.")
    print("Por favor, instale-a com: pip install scikit-learn")
    sys.exit(1)

ARQUIVO_ENTRADA = 'data/genes_export.csv' 
K_VIZINHOS = 5
KMER_SIZE = 6
N_CLUSTERS = 200  
TEST_SET_SIZE = 0.2 
PERCENT_FALTANTE = 0.05 

def distancia_hamming(s1, s2):
    if len(s1) != len(s2):
        return abs(len(s1) - len(s2)) + max(len(s1), len(s2)) // 2
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def voto_majoritario(vizinhos, pos_i):
    bases = [seq[pos_i] for seq in vizinhos if len(seq) > pos_i and seq[pos_i] != 'N']
    if not bases:
        return 'N'
    vencedor = Counter(bases).most_common(1)[0][0]
    return vencedor

def get_kmers_string(sequence, k):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return " ".join(kmers)

def rodar_validacao_hibrida(kmer_size, n_clusters, k_vizinhos):
    print(f"Iniciando VALIDAÇÃO (Clusterização + KNN Simples)...")
    print(f"Parâmetros: k-mer size={kmer_size}, N Clusters={n_clusters}, k-NN={k_vizinhos}\n")

    print(f"Carregando dados de {ARQUIVO_ENTRADA}...")
    try:
        df = pd.read_csv(ARQUIVO_ENTRADA) 
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
        print(f"Verifique se o caminho está correto (esperando: {os.path.abspath(ARQUIVO_ENTRADA)})")
        return False

    df['Sequencia'] = df['Sequencia'].astype(str).fillna('')
    df_limpo = df[~df['Sequencia'].str.contains('N')].copy()

    if len(df_limpo) < 1000:
        print(f"ERRO: Dados limpos insuficientes ({len(df_limpo)}) para clusterização.")
        return False

    print(f"Total de {len(df_limpo)} sequências limpas encontradas.")

    print(f"Iniciando engenharia de features (k-mers CountVec, k={kmer_size})...")
    df_limpo['kmers'] = df_limpo['Sequencia'].apply(lambda x: get_kmers_string(x, kmer_size))

    vectorizer = CountVectorizer(analyzer='word')
    kmer_vectors = vectorizer.fit_transform(df_limpo['kmers'])
    print(f"Dataset transformado em uma matriz de {kmer_vectors.shape[0]} genes e {kmer_vectors.shape[1]} features.")

    print(f"Iniciando clusterização K-Means (N Clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) 
    df_limpo['Cluster'] = kmeans.fit_predict(kmer_vectors)
    print("Clusterização concluída. Genes foram atribuídos a 'famílias'.\n")

    print(f"Dividindo dados: {1-TEST_SET_SIZE:.0%} para Referência, {TEST_SET_SIZE:.0%} para Validação.")
    df_referencia, df_teste_original = train_test_split(
        df_limpo, 
        test_size=TEST_SET_SIZE, 
        random_state=42
    )

    print(f"Total de vizinhos para referência: {len(df_referencia)}")
    print(f"Total de sequências para teste (gabarito): {len(df_teste_original)}")

    y_true = []
    y_pred = []
    print(f"\nIniciando simulação... Serão 'furados' {PERCENT_FALTANTE:.0%} de cada gene de teste.")

    for index, linha in df_teste_original.iterrows():
        seq_original = linha['Sequencia']
        seq_id = linha.iloc[0] 
        gene_cluster = linha['Cluster']

        seq_poked_list = list(seq_original)
        n_holes = int(len(seq_original) * PERCENT_FALTANTE)
        if n_holes == 0: continue
        
        try:
            indices_poked = random.sample(range(len(seq_original)), n_holes)
        except ValueError:
            continue
            
        for pos in indices_poked:
            y_true.append(seq_original[pos])
            seq_poked_list[pos] = 'N'
        seq_poked_str = "".join(seq_poked_list)

        df_local_referencia = df_referencia[df_referencia['Cluster'] == gene_cluster]
        
        if len(df_local_referencia) < k_vizinhos:
            lista_busca_seqs = df_referencia['Sequencia'].tolist()
        else:
            lista_busca_seqs = df_local_referencia['Sequencia'].tolist()

        distancias = []
        for ref_seq in lista_busca_seqs:
            dist = distancia_hamming(seq_poked_str, ref_seq)
            distancias.append(dist)
        
        indices_vizinhos = np.argsort(distancias)[:k_vizinhos]
        vizinhos_seqs = [lista_busca_seqs[i] for i in indices_vizinhos]

        for pos in indices_poked:
            base_imputada = voto_majoritario(vizinhos_seqs, pos)
            y_pred.append(base_imputada)

    print("\n--- Relatório de Desempenho (Clusterização + KNN Simples) ---")
    if not y_true:
        print("Nenhuma base foi imputada. O script pode ter sido interrompido.")
        return False

    labels = sorted(list(set(y_true)))
    acc = accuracy_score(y_true, y_pred)
    print(f"Acurácia Geral: {acc * 100:.2f}%")
    print(f"(De {len(y_true)} bases 'N' simuladas, o KNN acertou a base correta em {acc*len(y_true):.0f} casos.)\n")

    print("\nRelatório de Classificação (Precisão, Recall, F1 por Base):")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("Validação concluída.")
    return True

if __name__ == "__main__":
    rodar_validacao_hibrida(
        kmer_size=KMER_SIZE, 
        n_clusters=N_CLUSTERS, 
        k_vizinhos=K_VIZINHOS
    )