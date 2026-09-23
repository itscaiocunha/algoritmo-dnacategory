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
    
    # --- NOVO IMPORT ---
    from imblearn.over_sampling import SMOTE 
    
except ImportError:
    print("ERRO: Bibliotecas não encontradas.")
    print("Rode: pip install scikit-learn imbalanced-learn pandas numpy")
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
print(f"Iniciando Protótipo de Classificação (SVM com SMOTE Oversampling)...")
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

# --- 3.3 Achatamento (Handling Duplicates) ---
print("Achatando dataset (1 linha por gene)...")
df_labels_flat = df.groupby('join_key')['label'].max()
df_seq_flat = df.groupby('join_key')['Sequencia'].first()
df_flat = pd.merge(df_seq_flat, df_labels_flat, on='join_key', how='inner')

print(f"Dataset achatado para {len(df_flat)} genes únicos.")
print(f"Distribuição ANTES do balanceamento:\n{df_flat['label'].value_counts()}\n")

if df_flat['label'].nunique() < 2:
    print("\nERRO: Dataset não contém as duas classes. Saindo.")
    sys.exit(1)
    
# --- 3.4 Engenharia de Features (X) ---
# (Precisamos fazer isso ANTES do SMOTE)
print(f"Iniciando engenharia de features (k-mers, k={KMER_SIZE})...")
vectorizer = CountVectorizer(analyzer='word')
X_kmers = vectorizer.fit_transform(df_flat['Sequencia'].apply(lambda x: get_kmers_string(x, KMER_SIZE)))
Y_labels = df_flat['label']

print(f"Matriz de features X criada: {X_kmers.shape}")
print(f"Vetor de labels Y criado: {Y_labels.shape}")

# --- 3.5 Divisão de Treino/Teste ---
# (CRUCIAL: Dividimos ANTES de aplicar o SMOTE)
print("\nDividindo dados (80% treino / 20% teste)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_kmers, 
    Y_labels, 
    test_size=0.2, 
    random_state=42,
    stratify=Y_labels # Garante que a proporção 4:1 vá para o teste
)
print(f"Tamanho do treino (antes do SMOTE): {X_train.shape[0]}")
print(f"Tamanho do teste: {X_test.shape[0]}")

# --- 3.6 Balanceamento com SMOTE (Apenas nos dados de TREINO) ---
print("\nAplicando SMOTE Oversampling nos dados de TREINO...")
# (Nunca aplicamos SMOTE nos dados de teste, pois queremos testar no mundo real)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Tamanho do treino (DEPOIS do SMOTE): {X_train_balanced.shape[0]}")

# --- 3.7 Treinamento do SVM ---
print("Iniciando treinamento do SVM (kernel='linear')...")
# Não precisamos mais do class_weight='balanced'
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_balanced, y_train_balanced)

print("Treinamento concluído.")

# --- 3.8 Avaliação (Nos dados de teste DESBALANCEADOS) ---
print("\nAvaliando modelo nos dados de teste (do mundo real)...")
y_pred = svm_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n--- Relatório de Desempenho (Protótipo SVM + SMOTE) ---")
print(f"Acurácia Geral: {acc * 100:.2f}%")

print("\nRelatório de Classificação (Precisão, Recall, F1 por Classe):")
print(classification_report(y_test, y_pred))

print("Protótipo de classificação concluído.")