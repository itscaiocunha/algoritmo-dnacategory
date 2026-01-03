import pandas as pd
import numpy as np
from collections import Counter
import sys
import os
import joblib 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

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
PATH_MESTRE_LIMPO = os.path.join(PATH_DADOS, 'dataset_mestre_IMPUTADO_FINAL.csv')

# --- NOVO ARQUIVO DE "PAUSA" ---
PATH_PROGRESSO = os.path.join(PATH_DADOS, 'imputacao_progresso.txt')

# --- Parâmetros do Modelo ---
KMER_SIZE = 6
K_VIZINHOS = 5
CHUNK_SIZE = 5000

# --- 2. Funções Auxiliares ---
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

# --- 3. NOVAS Funções de "Pausa" (Checkpointing) ---
def carregar_progresso(caminho_progresso):
    """Lê o arquivo de progresso e retorna o último chunk concluído."""
    try:
        with open(caminho_progresso, 'r') as f:
            progresso = int(f.read().strip())
            return progresso
    except FileNotFoundError:
        return 0 # Começa do zero
    except ValueError:
        return 0 # Arquivo corrompido, começa do zero

def salvar_progresso(caminho_progresso, chunk_index):
    """Salva o índice do próximo chunk a ser processado."""
    try:
        with open(caminho_progresso, 'w') as f:
            f.write(str(chunk_index + 1)) # Salva o *próximo* índice
    except IOError as e:
        print(f"!!! ATENÇÃO: Falha ao salvar progresso: {e} !!!")


# --- 4. Função Principal de Processamento (O Coração) ---
# (Esta função não muda)
def processar_linha(linha, vectorizer, kmeans, df_referencia, kmer_size, k_vizinhos):
    sequencia_suja = linha['Sequencia']
    join_key = linha['join_key']
    
    if 'N' not in str(sequencia_suja):
        return linha 

    nova_sequencia = list(sequencia_suja)
    indices_n = [i for i, base in enumerate(nova_sequencia) if base == 'N']
    if not indices_n:
        return linha

    try:
        kmers_str = get_kmers_string(sequencia_suja, kmer_size)
        vetor_kmer = vectorizer.transform([kmers_str])
        cluster_id = kmeans.predict(vetor_kmer)[0]

        df_vizinhos_cluster = df_referencia[df_referencia['Cluster'] == cluster_id]

        if len(df_vizinhos_cluster) < k_vizinhos:
            df_vizinhos_cluster = df_referencia.sample(n=k_vizinhos*2, random_state=42)

        distancias = []
        for ref_seq in df_vizinhos_cluster['Sequencia']:
            dist = distancia_hamming(sequencia_suja, ref_seq)
            distancias.append(dist)
        
        indices_vizinhos = np.argsort(distancias)[:k_vizinhos]
        seqs_vencedoras = df_vizinhos_cluster.iloc[indices_vizinhos]['Sequencia'].tolist()
        dists_vencedoras = [distancias[i] for i in indices_vizinhos]
        
        n_imputados = 0
        for i in indices_n:
            base_imputada = voto_majoritario_ponderado(seqs_vencedoras, dists_vencedoras, i)
            if base_imputada != 'N':
                nova_sequencia[i] = base_imputada
                n_imputados += 1
        
        linha['Sequencia'] = "".join(nova_sequencia) 
        print(f"  -> ID {join_key} (Cluster {cluster_id}): {n_imputados} bases 'N' imputadas.")
        
    except Exception as e:
        print(f"  -> ERRO processando {join_key}: {e}. Pulando linha.")
        
    return linha

# --- 5. Bloco de Execução Principal (O Loop de Chunks) ---
if __name__ == "__main__":
    
    print("Iniciando Pipeline de Imputação (COM CHECKPOINTS)...")
    
    # 1. Carregar os modelos e a referência
    print("Carregando modelos (Vectorizer, KMeans)...")
    try:
        vectorizer = joblib.load(PATH_VECTORIZER)
        kmeans = joblib.load(PATH_KMEANS)
    except FileNotFoundError:
        print(f"ERRO: Modelos não encontrados. Rode 'treinar_imputador.py' primeiro.")
        sys.exit(1)
        
    print("Carregando 'Lista Telefônica' de referência...")
    try:
        df_referencia = pd.read_csv(PATH_REFERENCIA)
        print("Referência carregada na memória.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{PATH_REFERENCIA}' não encontrado. Rode 'treinar_imputador.py'.")
        sys.exit(1)

    # --- LÓGICA DE PAUSA MODIFICADA ---
    chunks_completos = carregar_progresso(PATH_PROGRESSO)
    
    if chunks_completos == 0:
        print("Progresso não encontrado. Começando do zero.")
        # Se estamos começando do zero, apague o arquivo antigo
        if os.path.exists(PATH_MESTRE_LIMPO):
            os.remove(PATH_MESTRE_LIMPO)
        escrever_header = True
        modo_escrita = 'w' # Modo 'write' (escrever novo)
    else:
        print(f"Progresso encontrado. Resumindo a partir do chunk {chunks_completos}.")
        escrever_header = False
        modo_escrita = 'a' # Modo 'append' (anexar)
    # --- FIM DA LÓGICA DE PAUSA ---

    print("Modelos e Referência carregados. Iniciando processamento do arquivo...")
    
    try:
        reader = pd.read_csv(PATH_MESTRE_SUJO, chunksize=CHUNK_SIZE)
        
        for i, chunk in enumerate(reader):
            
            # --- PULAR CHUNKS JÁ FEITOS ---
            if i < chunks_completos:
                print(f"Pulando Pedaço (Chunk) {i}... (já processado)")
                continue
            # --- FIM DO PULO ---
            
            print(f"\n--- Processando Pedaço (Chunk) {i} (Linhas {i*CHUNK_SIZE} a {(i+1)*CHUNK_SIZE}) ---")
            
            if 'join_key' not in chunk.columns:
                 chunk['join_key'] = chunk.iloc[:, 0].apply(limpar_id)
            
            chunk_limpo = chunk.apply(
                processar_linha, 
                axis=1, 
                args=(vectorizer, kmeans, df_referencia, KMER_SIZE, K_VIZINHOS)
            )
            
            # Salvar o pedaço limpo no disco
            chunk_limpo.to_csv(PATH_MESTRE_LIMPO, mode=modo_escrita, index=False, header=escrever_header)
            
            # --- ATUALIZAR O "MARCADOR DE PÁGINA" ---
            salvar_progresso(PATH_PROGRESSO, i)
            
            # Desativa o header e muda para 'append' após o primeiro chunk (caso tenha começado do zero)
            escrever_header = False
            modo_escrita = 'a'
                
            print(f"--- Pedaço {i} salvo e progresso '{i+1}' registrado. ---")

    except FileNotFoundError:
        print(f"ERRO: Arquivo mestre '{PATH_MESTRE_SUJO}' não encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"ERRO INESPERADO DURANTE O PROCESSAMENTO: {e}")
        
    print("\n--- Processamento em Lote Concluído ---")
    print(f"Arquivo final (imputado) salvo em: {PATH_MESTRE_LIMPO}")

    if os.path.exists(PATH_PROGRESSO):
        os.remove(PATH_PROGRESSO)