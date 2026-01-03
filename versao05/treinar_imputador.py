import pandas as pd
import sys
import os
import joblib 

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.cluster import KMeans
except ImportError:
    print("ERRO: scikit-learn não encontrado. Rode: pip install scikit-learn")
    sys.exit(1)

# --- Configurações ---
ARQUIVO_ENTRADA = '../data/genes_export.csv' 
PATH_MODELOS = '../modelos/' 
PATH_REFERENCIA = '../data/referencia_limpa_com_clusters.csv'

KMER_SIZE = 6
N_CLUSTERS = 200

# --- NOVA FUNÇÃO DE LIMPEZA DE ID ---
def limpar_id(id_sujo):
    """Limpa um ID de qualquer formato (com | ou .) para o formato ENSG..."""
    try:
        id_str = str(id_sujo)
        id_sem_pipe = id_str.split('|')[0].split()[0]
        id_sem_versao = id_sem_pipe.split('.')[0]
        if id_sem_versao.startswith('ENSG'):
            return id_sem_versao
        return None 
    except:
        return None

# --- Função de k-mer (a mesma de antes) ---
def get_kmers_string(sequence, k):
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return " ".join(kmers)

# --- 1. Carregar Dados Limpos ---
print(f"Carregando dados limpos de {ARQUIVO_ENTRADA}...")
try:
    df = pd.read_csv(ARQUIVO_ENTRADA)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
    sys.exit(1)

df['Sequencia'] = df['Sequencia'].astype(str).fillna('')
df_limpo = df[~df['Sequencia'].str.contains('N')].copy()

if len(df_limpo) < 1000:
    print("ERRO: Dados limpos insuficientes para treinar.")
    sys.exit(1)

print(f"Total de {len(df_limpo)} sequências limpas para treinamento.")

# --- 2. Engenharia de Features (k-mers) ---
print(f"Iniciando engenharia de features (k-mers CountVec, k={KMER_SIZE})...")
df_limpo['kmers'] = df_limpo['Sequencia'].apply(lambda x: get_kmers_string(x, KMER_SIZE))

vectorizer = CountVectorizer(analyzer='word')
kmer_vectors = vectorizer.fit_transform(df_limpo['kmers'])
print(f"Matriz de features criada: {kmer_vectors.shape}")

# --- 3. Treinamento do Cluster (K-Means) ---
print(f"Iniciando treinamento do K-Means (N Clusters={N_CLUSTERS})...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10) 
df_limpo['Cluster'] = kmeans.fit_predict(kmer_vectors)
print("Treinamento K-Means concluído.")

# --- 4. Salvar Modelos no Disco ---
os.makedirs(PATH_MODELOS, exist_ok=True)
print(f"Salvando modelos em '{PATH_MODELOS}'...")

path_vec = os.path.join(PATH_MODELOS, 'count_vectorizer_k6.joblib')
joblib.dump(vectorizer, path_vec)
print(f"Modelo CountVectorizer salvo em: {path_vec}")

path_kmeans = os.path.join(PATH_MODELOS, 'kmeans_n200_k6.joblib')
joblib.dump(kmeans, path_kmeans)
print(f"Modelo K-Means salvo em: {path_kmeans}")

# --- 5. Salvar a "Lista Telefônica" de Vizinhos ---
print("Criando arquivo de referência de vizinhos...")

# --- CORREÇÃO AQUI ---
# 1. Crie a coluna 'join_key' limpando a primeira coluna (que tem o ID sujo)
df_limpo['join_key'] = df_limpo.iloc[:, 0].apply(limpar_id)

# 2. Agora sim, selecione as colunas limpas
df_referencia = df_limpo[['join_key', 'Sequencia', 'Cluster']] 
df_referencia = df_referencia.dropna(subset=['join_key']) # Garante que não há IDs nulos

df_referencia.to_csv(PATH_REFERENCIA, index=False)
print(f"Arquivo de referência de vizinhos salvo em: {PATH_REFERENCIA}")

print("\n--- Treinamento Concluído ---")