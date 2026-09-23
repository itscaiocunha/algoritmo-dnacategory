import pandas as pd
import numpy as np
import sys
import os

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.exceptions import UndefinedMetricWarning
    import warnings
    from imblearn.over_sampling import SMOTE
    
except ImportError:
    print("ERRO: Bibliotecas não encontradas.")
    print("Rode: pip install scikit-learn imbalanced-learn pandas numpy")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- 1. Configurações ---
ARQUIVO_ENTRADA = '../../data/dataset_mestre_IMPUTADO_FINAL.csv'

KMER_SIZE = 6 
TARGET_FUNCTION = 'protein binding'

# --- 2. Função de k-mer ---
def get_kmers_string(sequence, k):
    seq_str = str(sequence)
    kmers = [seq_str[i:i+k] for i in range(len(seq_str) - k + 1)]
    return " ".join(kmers)

# --- 3. Script Principal ---
print(f"Iniciando Treinamento FINAL (SVM com Undersampling)...")
print(f"Arquivo de entrada: {ARQUIVO_ENTRADA}")
print(f"Tarefa: Classificação Binária (Target = '{TARGET_FUNCTION}')\n")

# --- 3.1 Carregar Dados ---
print("Carregando dataset mestre (14.4M+ linhas)... Isso pode levar alguns minutos.")
try:
    df = pd.read_csv(ARQUIVO_ENTRADA)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
    sys.exit(1)
print("Dataset carregado.")

if 'GO term name' not in df.columns or 'Sequencia' not in df.columns:
    print(f"ERRO: O CSV de entrada não contém 'GO term name' ou 'Sequencia'.")
    sys.exit(1)

# --- 3.2 Preparação dos Labels (Y) ---
print(f"Preparando labels... (1 = '{TARGET_FUNCTION}', 0 = Outro)")
df['label'] = np.where(df['GO term name'] == TARGET_FUNCTION, 1, 0)

# --- 3.3 Achatamento e BALANCEAMENTO ---
print("Achatando dataset (1 linha por gene)... Isso também pode demorar.")
df_labels_flat = df.groupby('join_key')['label'].max()
df_seq_flat = df.groupby('join_key')['Sequencia'].first()
df_flat = pd.merge(df_seq_flat, df_labels_flat, on='join_key', how='inner')

print(f"Dataset achatado para {len(df_flat)} genes únicos.")
print(f"Distribuição ANTES do balanceamento:\n{df_flat['label'].value_counts()}\n")

if df_flat['label'].nunique() < 2:
    print("\nERRO: Dataset não contém as duas classes. Saindo.")
    sys.exit(1)

# --- Aplicando UNDERSAMPLING (Nossa melhor estratégia) ---
print("Iniciando Undersampling para forçar balanço 1:1...")
df_majority = df_flat[df_flat['label'] == 1]
df_minority = df_flat[df_flat['label'] == 0]
n_minority = len(df_minority)

print(f"Classe Rara (0) tem {n_minority} amostras.")
print(f"Classe Comum (1) tem {len(df_majority)} amostras.")

df_majority_downsampled = df_majority.sample(n=n_minority, random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])

print(f"Dataset balanceado criado.")
print(f"Distribuição DEPOIS do balanceamento:\n{df_balanced['label'].value_counts()}\n")

# --- 3.4 Engenharia de Features (X) ---
print(f"Iniciando engenharia de features (k-mers, k={KMER_SIZE})...")
vectorizer = CountVectorizer(analyzer='word')

# Treinamos o Vectorizer em TODOS os dados achatados
vectorizer.fit(df_flat['Sequencia'].apply(lambda x: get_kmers_string(x, KMER_SIZE)))
# Transformamos apenas os dados balanceados
X_kmers = vectorizer.transform(df_balanced['Sequencia'].apply(lambda x: get_kmers_string(x, KMER_SIZE)))
Y_labels = df_balanced['label']

print(f"Matriz de features X criada: {X_kmers.shape}")
print(f"Vetor de labels Y criado: {Y_labels.shape}")

# --- 3.5 Treinamento do SVM ---
print("\nDividindo dados (80% treino / 20% teste)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_kmers, 
    Y_labels, 
    test_size=0.2, 
    random_state=42,
    stratify=Y_labels 
)

print(f"Iniciando treinamento do SVM (kernel='linear') em {len(y_train)} amostras...")
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train) 
print("Treinamento concluído.")

# --- 3.6 Avaliação Final ---
print("\nAvaliando modelo nos dados de teste...")
y_pred = svm_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n--- Relatório de Desempenho (SVM FINAL Balanceado) ---")
print(f"Acurácia Geral: {acc * 100:.2f}%")

print("\nRelatório de Classificação (Precisão, Recall, F1 por Classe):")
print(classification_report(y_test, y_pred))

print("\n\nTreinamento e Avaliação do Modelo Final CONCLUÍDOS.")