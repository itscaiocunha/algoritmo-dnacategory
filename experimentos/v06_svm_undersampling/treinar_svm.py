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
except ImportError:
    print("ERRO: scikit-learn não encontrado. Rode: pip install scikit-learn")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- 1. Configurações ---
ARQUIVO_ENTRADA = '../../data/dataset_mestre_IMPUTADO.csv' # O de 100k linhas
KMER_SIZE = 6 
TARGET_FUNCTION = 'protein binding'

# --- 2. Função de k-mer ---
def get_kmers_string(sequence, k):
    seq_str = str(sequence)
    kmers = [seq_str[i:i+k] for i in range(len(seq_str) - k + 1)]
    return " ".join(kmers)

# --- 3. Script Principal ---
print(f"Iniciando Protótipo de Classificação (SVM com Undersampling)...")
print(f"Arquivo de entrada: {ARQUIVO_ENTRADA}")
print(f"Tarefa: Classificação Binária (Target = '{TARGET_FUNCTION}')\n")

# --- 3.1 Carregar Dados ---
try:
    df = pd.read_csv(ARQUIVO_ENTRADA)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
    sys.exit(1)

if 'GO term name' not in df.columns or 'Sequencia' not in df.columns:
    print(f"ERRO: O CSV de entrada não contém 'GO term name' ou 'Sequencia'.")
    sys.exit(1)

# --- 3.2 Preparação dos Labels (Y) ---
print(f"Preparando labels... (1 = '{TARGET_FUNCTION}', 0 = Outro)")
df['label'] = np.where(df['GO term name'] == TARGET_FUNCTION, 1, 0)

# --- 3.3 Achatamento e BALANCEAMENTO ---
print("Achatando dataset (1 linha por gene)...")
df_labels_flat = df.groupby('join_key')['label'].max()
df_seq_flat = df.groupby('join_key')['Sequencia'].first()
df_flat = pd.merge(df_seq_flat, df_labels_flat, on='join_key', how='inner')

print(f"Dataset achatado para {len(df_flat)} genes únicos.")
print(f"Distribuição ANTES do balanceamento:\n{df_flat['label'].value_counts()}\n")

if df_flat['label'].nunique() < 2:
    print("\nERRO: Dataset não contém as duas classes. Saindo.")
    sys.exit(1)

# --- INÍCIO DA NOVA ETAPA: UNDERSAMPLING ---
print("Iniciando Undersampling para forçar balanço 1:1...")

# Separa as classes
df_majority = df_flat[df_flat['label'] == 1]
df_minority = df_flat[df_flat['label'] == 0]

# Pega o número de amostras da classe rara
n_minority = len(df_minority)

# Seleciona aleatoriamente o 'n_minority' amostras da classe comum
df_majority_downsampled = df_majority.sample(n=n_minority, random_state=42)

# Junta as duas de volta
df_balanced = pd.concat([df_majority_downsampled, df_minority])

print(f"Dataset balanceado criado (180 vs 180).")
print(f"Distribuição DEPOIS do balanceamento:\n{df_balanced['label'].value_counts()}\n")
# --- FIM DA NOVA ETAPA ---


# --- 3.4 Engenharia de Features (X) ---
print(f"Iniciando engenharia de features (k-mers, k={KMER_SIZE})...")
vectorizer = CountVectorizer(analyzer='word')

# *** MUITO IMPORTANTE: Treinar o Vectorizer no 'df_flat' (todos os dados) ***
# Isso garante que o vocabulário (as 4110 features) inclua
# k-mers dos dados que jogamos fora, o que é bom.
vectorizer.fit(df_flat['Sequencia'].apply(lambda x: get_kmers_string(x, KMER_SIZE)))

# ... mas transformar apenas os dados balanceados
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
    stratify=Y_labels # Garante 50/50 no treino e teste
)

print("Iniciando treinamento do SVM (kernel='linear')...")
# Agora não precisamos mais do class_weight='balanced', pois os dados JÁ são balanceados
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

print("Treinamento concluído.")

# --- 3.6 Avaliação do Protótipo ---
print("\nAvaliando modelo nos dados de teste...")
y_pred = svm_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n--- Relatório de Desempenho (Protótipo SVM Balanceado) ---")
print(f"Acurácia Geral: {acc * 100:.2f}%")
print(f"(Acurácia em um dataset 1:1. Chutar '1' agora só daria 50%.)")

print("\nRelatório de Classificação (Precisão, Recall, F1 por Classe):")
print(classification_report(y_test, y_pred))

print("Protótipo de classificação concluído.")