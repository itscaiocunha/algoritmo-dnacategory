import pandas as pd
import numpy as np
from collections import Counter
import sys
import os
import joblib 

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances # Usaremos para Dist. Hamming
except ImportError:
    print("ERRO: scikit-learn não encontrado. Rode: pip install scikit-learn")
    sys.exit(1)

# --- 1. Configurações e Caminhos ---
PATH_MODELOS = '../modelos/'
PATH_DADOS = '../data/'

# --- ARQUIVOS DE ENTRADA (Nossas Ferramentas) ---
PATH_VECTORIZER = os.path.join(PATH_MODELOS, 'count_vectorizer_k6.joblib')
PATH_KMEANS = os.path.join(PATH_MODELOS, 'kmeans_n200_k6.joblib')
PATH_REFERENCIA = os.path.join(PATH_DADOS, 'referencia_limpa_com_clusters.csv')

# --- ARQUIVO "MONSTRO" DE ENTRADA ---
PATH_MESTRE_SUJO = os.path.join(PATH_DADOS, 'dataset_mestre.csv')

# --- ARQUIVO FINAL DE SAÍDA ---
PATH_MESTRE_LIMPO = os.path.join(PATH_DADOS, 'dataset_mestre_IMPUTADO.csv')

# --- Parâmetros do Modelo (DEVEM ser iguais aos do treino) ---
KMER_SIZE = 6
K_VIZINHOS = 5
CHUNK_SIZE = 5000

# --- 2. Funções Auxiliares (As mesmas de antes) ---
def limpar_id(id_sujo):
    try:
        id_str = str(id_sujo)
        id_sem_pipe = id_str.split('|')[0].split()[0]
        id_sem_versao = id_sem_pipe.split('.')[0]
        if id_sem_versao.startswith('ENSG'):
            return id_sem_versao
        return None 
    except:
        return None

def get_kmers_string(sequence, k):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return " ".join(kmers)

def distancia_hamming(s1, s2):
    if len(s1) != len(s2):
        return abs(len(s1) - len(s2)) + max(len(s1), len(s2)) // 2
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def voto_majoritario_ponderado(vizinhos_seqs, vizinhos_dist, pos_i):
    votos = {'A': 0, 'T': 0, 'C': 0, 'G': 0}
    for seq, dist in zip(vizinhos_seqs, vizinhos_dist):
        if len(seq) > pos_i and seq[pos_i] in votos:
            base = seq[pos_i]
            peso = 1.0 / (dist + 1.0) 
            votos[base] += peso
    
    votos_validos = {base: peso for base, peso in votos.items() if peso > 0}
    if not votos_validos:
        return 'N'
    return max(votos_validos, key=votos_validos.get)

# --- 3. Função Principal de Processamento (O Coração) ---
def processar_linha(linha, vectorizer, kmeans, df_referencia, kmer_size, k_vizinhos):
    """
    Aplica o pipeline de imputação completo em uma ÚNICA linha (DataFrame Series).
    """
    # Pega a sequência (X) e o ID (chave)
    sequencia_suja = linha['Sequencia']
    join_key = linha['join_key'] # Assume que a coluna 'join_key' já existe
    
    # 1. Se não tiver 'N', não faça nada. Economize tempo.
    if 'N' not in str(sequencia_suja):
        return linha # Retorna a linha original

    # 2. Se tiver 'N', prepare para imputar
    nova_sequencia = list(sequencia_suja)
    indices_n = [i for i, base in enumerate(nova_sequencia) if base == 'N']
    if not indices_n:
        return linha # Segurança

    try:
        # 3. Descobrir o Cluster (Usando os "cérebros" carregados)
        kmers_str = get_kmers_string(sequencia_suja, kmer_size)
        vetor_kmer = vectorizer.transform([kmers_str])
        cluster_id = kmeans.predict(vetor_kmer)[0]

        # 4. Encontrar Vizinhos (Usando a "lista telefônica")
        df_vizinhos_cluster = df_referencia[df_referencia['Cluster'] == cluster_id]

        if len(df_vizinhos_cluster) < k_vizinhos:
            # Fallback: Cluster muito pequeno, usa a referência global
            df_vizinhos_cluster = df_referencia.sample(n=k_vizinhos*2, random_state=42) # Pega uma amostra

        # 5. Calcular Distâncias (A parte lenta)
        distancias = []
        for ref_seq in df_vizinhos_cluster['Sequencia']:
            dist = distancia_hamming(sequencia_suja, ref_seq)
            distancias.append(dist)
        
        # 6. Pegar os Top-K e seus dados
        indices_vizinhos = np.argsort(distancias)[:k_vizinhos]
        seqs_vencedoras = df_vizinhos_cluster.iloc[indices_vizinhos]['Sequencia'].tolist()
        dists_vencedoras = [distancias[i] for i in indices_vizinhos]
        
        # 7. Imputar (Voto Ponderado)
        n_imputados = 0
        for i in indices_n:
            base_imputada = voto_majoritario_ponderado(seqs_vencedoras, dists_vencedoras, i)
            if base_imputada != 'N':
                nova_sequencia[i] = base_imputada
                n_imputados += 1
        
        # 8. Atualizar a linha do DataFrame
        linha['Sequencia'] = "".join(nova_sequencia) 
        print(f"  -> ID {join_key} (Cluster {cluster_id}): {n_imputados} bases 'N' imputadas.")
        
    except Exception as e:
        print(f"  -> ERRO processando {join_key}: {e}. Pulando linha.")
        
    return linha

# --- 4. Bloco de Execução Principal (O Loop de Chunks) ---
if __name__ == "__main__":
    
    print("Iniciando Pipeline de Imputação em Lote (Produção)...")
    
    # 1. Carregar os modelos e a referência (só uma vez)
    print("Carregando modelos (Vectorizer, KMeans) da pasta 'modelos/'...")
    try:
        vectorizer = joblib.load(PATH_VECTORIZER)
        kmeans = joblib.load(PATH_KMEANS)
    except FileNotFoundError:
        print(f"ERRO: Modelos não encontrados. Rode 'treinar_imputador.py' primeiro.")
        sys.exit(1)
        
    print("Carregando 'Lista Telefônica' de referência (77k linhas)...")
    try:
        df_referencia = pd.read_csv(PATH_REFERENCIA)
        # Convertemos para listas para acesso mais rápido (se couber na RAM)
        # (Otimização: Se a RAM for um problema, deixe como dataframe)
        print("Referência carregada na memória.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{PATH_REFERENCIA}' não encontrado. Rode 'treinar_imputador.py'.")
        sys.exit(1)

    print("Modelos e Referência carregados. Iniciando processamento do arquivo de 20GB...")
    
    # 2. Processar o arquivo de 20GB em pedaços
    
    # Apaga o arquivo de saída antigo, se existir
    if os.path.exists(PATH_MESTRE_LIMPO):
        os.remove(PATH_MESTRE_LIMPO)
        
    is_first_chunk = True
    
    try:
        reader = pd.read_csv(PATH_MESTRE_SUJO, chunksize=CHUNK_SIZE)
        
        for i, chunk in enumerate(reader):
            print(f"\n--- Processando Pedaço (Chunk) {i} (Linhas {i*CHUNK_SIZE} a {(i+1)*CHUNK_SIZE}) ---")
            
            # 3. Limpar IDs no Pedaço
            # (Assume que a coluna de ID sujo é a primeira ou 'join_key')
            if 'join_key' not in chunk.columns:
                 # Se o 'juntar_dados' não salvou a 'join_key', crie-a
                 chunk['join_key'] = chunk.iloc[:, 0].apply(limpar_id)
            
            # 4. Aplicar a imputação (linha por linha)
            # axis=1 significa "aplicar a função em cada linha"
            chunk_limpo = chunk.apply(
                processar_linha, 
                axis=1, 
                args=(vectorizer, kmeans, df_referencia, KMER_SIZE, K_VIZINHOS)
            )
            
            # 5. Salvar o pedaço limpo no disco
            if is_first_chunk:
                # Salva com cabeçalho
                chunk_limpo.to_csv(PATH_MESTRE_LIMPO, mode='a', index=False, header=True)
                is_first_chunk = False
            else:
                # Salva sem cabeçalho (apenas anexa)
                chunk_limpo.to_csv(PATH_MESTRE_LIMPO, mode='a', index=False, header=False)
                
            print(f"--- Pedaço {i} salvo em {PATH_MESTRE_LIMPO} ---")

            if i == 19:
                print(f"\n*** MODO DE TESTE: Interrompendo após {i+1} chunks (aprox. 25k linhas). ***")
                break

    except FileNotFoundError:
        print(f"ERRO: Arquivo mestre '{PATH_MESTRE_SUJO}' não encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"ERRO INESPERADO DURANTE O PROCESSAMENTO: {e}")
        
    print("\n--- Processamento em Lote Concluído ---")
    print(f"Arquivo final (imputado) salvo em: {PATH_MESTRE_LIMPO}")