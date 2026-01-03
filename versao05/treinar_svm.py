import pandas as pd
import numpy as np
import sys
import os

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC # Support Vector Classifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.exceptions import UndefinedMetricWarning
    import warnings
except ImportError:
    print("ERRO: scikit-learn não encontrado. Rode: pip install scikit-learn")
    sys.exit(1)

# Ignora warnings de "Recall é 0" (normal em datasets pequenos/desbalanceados)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- 1. Configurações ---
ARQUIVO_ENTRADA = '../data/dataset_mestre_IMPUTADO.csv' # O de 5000 linhas
KMER_SIZE = 6 # Deve ser igual ao do imputador

# --- !! ESCOLHA SEU ALVO AQUI !! ---
# Mude isso para a função mais comum que você viu no seu 'juntar_dados.py'
TARGET_FUNCTION = 'protein binding'

# --- 2. Função de k-mer (a mesma de antes) ---
def get_kmers_string(sequence, k):
    """Converte 'ATGC' em 'ATG TGC' (para k=3)."""
    # Garante que a sequência é uma string
    seq_str = str(sequence)
    kmers = [seq_str[i:i+k] for i in range(len(seq_str) - k + 1)]
    return " ".join(kmers)

# --- 3. Script Principal ---
print(f"Iniciando Protótipo de Classificação (SVM)...")
print(f"Arquivo de entrada: {ARQUIVO_ENTRADA}")
print(f"Tarefa: Classificação Binária (Target = '{TARGET_FUNCTION}')\n")

# --- 3.1 Carregar Dados ---
try:
    df = pd.read_csv(ARQUIVO_ENTRADA)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
    print("Execute o 'aplicar_imputador.py' (em modo teste) primeiro.")
    sys.exit(1)

# Garante que as colunas corretas existem
if 'GO term name' not in df.columns or 'Sequencia' not in df.columns:
    print(f"ERRO: O CSV de entrada não contém 'GO term name' ou 'Sequencia'.")
    sys.exit(1)

# --- 3.2 Preparação dos Labels (Y) ---
print(f"Preparando labels... (1 = '{TARGET_FUNCTION}', 0 = Outro)")
# Cria o label binário
df['label'] = np.where(df['GO term name'] == TARGET_FUNCTION, 1, 0)

# --- 3.3 Achatamento (Handling Duplicates) ---
# Precisamos de 1 linha por gene.
# Se um gene é 'protein binding' E 'outra coisa', ele é 'protein binding' (usamos max()).
print("Achatando dataset (1 linha por gene)...")

# Primeiro, agrupa os dados para criar o 'Y' (labels)
# .max() garante que se *qualquer* anotação for o nosso alvo, o gene recebe 1
df_labels_flat = df.groupby('join_key')['label'].max()

# Depois, agrupa para pegar o 'X' (sequência)
# .first() pega a primeira ocorrência da sequência (elas são todas iguais)
df_seq_flat = df.groupby('join_key')['Sequencia'].first()

# Junta os dois de volta
df_flat = pd.merge(df_seq_flat, df_labels_flat, on='join_key', how='inner')

print(f"Dataset 'longo' (5000 linhas) achatado para {len(df_flat)} genes únicos.")

# Checa se temos as duas classes
if df_flat['label'].nunique() < 2:
    print(f"\nERRO: O dataset de teste não contém as duas classes (0 e 1).")
    print(f"Apenas {df_flat['label'].nunique()} classe foi encontrada.")
    print(f"Tente um 'TARGET_FUNCTION' diferente (mais comum) no topo do script.")
    sys.exit(1)

print(f"Distribuição das classes:\n{df_flat['label'].value_counts()}\n")

# --- 3.4 Engenharia de Features (X) ---
print(f"Iniciando engenharia de features (k-mers, k={KMER_SIZE})...")
vectorizer = CountVectorizer(analyzer='word')
X_kmers = vectorizer.fit_transform(df_flat['Sequencia'].apply(lambda x: get_kmers_string(x, KMER_SIZE)))
Y_labels = df_flat['label']

print(f"Matriz de features X criada: {X_kmers.shape}")
print(f"Vetor de labels Y criado: {Y_labels.shape}")

# --- 3.5 Treinamento do SVM ---
print("\nDividindo dados (80% treino / 20% teste)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_kmers, 
    Y_labels, 
    test_size=0.2, 
    random_state=42,
    stratify=Y_labels # Garante que as classes sejam balanceadas no split
)

print("Iniciando treinamento do SVM (kernel='linear')...")
# Usamos kernel='linear' pois é muito rápido e ótimo para dados de texto/k-mers (alta dimensão)
# C=1.0 é um bom padrão
svm_model = SVC(kernel='linear', C=1.0, random_state=42, class_weight='balanced')
svm_model.fit(X_train, y_train)

print("Treinamento concluído.")

# --- 3.6 Avaliação do Protótipo ---
print("\nAvaliando modelo nos dados de teste...")
y_pred = svm_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n--- Relatório de Desempenho (Protótipo SVM) ---")
print(f"Acurácia Geral: {acc * 100:.2f}%")
print(f"(Lembre-se: isso é treinado em dados de teste, a acurácia é apenas para debug)")

print("\nRelatório de Classificação (Precisão, Recall, F1 por Classe):")
print(classification_report(y_test, y_pred))

print("Protótipo de classificação concluído.")