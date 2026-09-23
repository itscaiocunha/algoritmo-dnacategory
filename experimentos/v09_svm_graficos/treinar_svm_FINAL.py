import pandas as pd
import numpy as np
import sys
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.exceptions import UndefinedMetricWarning
    from imblearn.under_sampling import RandomUnderSampler
except ImportError:
    print("ERRO: Bibliotecas não encontradas.")
    print("Rode: pip install scikit-learn imbalanced-learn pandas numpy tensorflow")
    sys.exit(1)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- 1. Configurações ---
ARQUIVO_ENTRADA = '../../data/dataset_FINAL_ACHATADO.csv' 
ARQUIVO_SAIDA_GRAFICOS = '../../reports/figures/'
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

# --- 3.1 Carregar Dados ---
print("Carregando dataset 'achatado'...")
try:
    df_flat = pd.read_csv(ARQUIVO_ENTRADA)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
    sys.exit(1)
print(f"Dataset carregado com {len(df_flat)} genes únicos.")

# --- 3.2 Balanceamento (Undersampling) ---
print(f"Distribuição ANTES do balanceamento:\n{df_flat['label'].value_counts()}\n")
print("Iniciando Undersampling para forçar balanço 1:1...")
rus = RandomUnderSampler(random_state=42)
X_para_amostrar = df_flat.index.values.reshape(-1, 1)
y_para_amostrar = df_flat['label']
X_res, y_res = rus.fit_resample(X_para_amostrar, y_para_amostrar)
indices_balanceados = X_res.flatten() 
df_balanced = df_flat.loc[indices_balanceados].copy()
print(f"Dataset balanceado criado.\nDistribuição DEPOIS:\n{df_balanced['label'].value_counts()}\n")

# --- 3.3 Engenharia de Features (X) ---
print(f"Iniciando engenharia de features (k-mers, k={KMER_SIZE})...")
vectorizer = CountVectorizer(analyzer='word')
print("Treinando Vectorizer (fit)...")
vectorizer.fit(df_flat['Sequencia'].astype(str).apply(lambda x: get_kmers_string(x, KMER_SIZE)))
print("Transformando dados balanceados (transform)...")
X_kmers = vectorizer.transform(df_balanced['Sequencia'].astype(str).apply(lambda x: get_kmers_string(x, KMER_SIZE)))
Y_labels = df_balanced['label']
print(f"Matriz de features X criada: {X_kmers.shape}")

# --- 3.4 Treinamento do SVM ---
print("\nDividindo dados (80% treino / 20% teste)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_kmers, 
    Y_labels, 
    test_size=0.2, 
    random_state=42,
    stratify=Y_labels 
)

print(f"Iniciando treinamento do SVM (kernel='linear') em {len(y_train)} amostras...")
# (Não precisamos de 'probability=True', usaremos 'decision_function' para o AUC)
svm_model = SVC(kernel='linear', C=1.0, random_state=42) 
svm_model.fit(X_train, y_train) 
print("Treinamento concluído.")

# --- 3.5 Avaliação Padrão ---
print("\nAvaliando modelo nos dados de teste...")
y_pred = svm_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n--- Relatório de Desempenho (SVM FINAL Balanceado) ---")
print(f"Acurácia Geral: {acc * 100:.2f}%")
print("\nRelatório de Classificação (Precisão, Recall, F1 por Classe):")
print(classification_report(y_test, y_pred))

# --- 3.6 NOVA ETAPA: Geração de Gráficos de Desempenho ---
print("\nGerando gráficos de desempenho...")
os.makedirs(ARQUIVO_SAIDA_GRAFICOS, exist_ok=True)

# --- MATRIZ DE CONFUSÃO ---
try:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Previsto: Outro (0)', 'Previsto: Alvo (1)'], 
                yticklabels=['Real: Outro (0)', 'Real: Alvo (1)'])
    plt.title(f'Matriz de Confusão - SVM (k=6, Undersampled)')
    plt.ylabel('Verdadeiro (Real)')
    plt.xlabel('Predito')
    path_matriz = os.path.join(ARQUIVO_SAIDA_GRAFICOS, 'matriz_confusao_svm.png')
    plt.savefig(path_matriz)
    print(f"Matriz de Confusão salva em: {path_matriz}")
    plt.close()
except Exception as e:
    print(f"Erro ao gerar Matriz de Confusão: {e}")

# --- CURVA ROC e AUC ---
try:
    # Para SVM linear, é melhor usar 'decision_function' do que 'predict_proba'
    y_scores = svm_model.decision_function(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:0.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Acaso (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC - SVM (k=6, Undersampled)')
    plt.legend(loc="lower right")
    path_roc = os.path.join(ARQUIVO_SAIDA_GRAFICOS, 'curva_roc_svm.png')
    plt.savefig(path_roc)
    print(f"Curva ROC salva em: {path_roc}")
    print(f"Score AUC (Área sob a Curva): {roc_auc:0.3f}")
    plt.close()
except Exception as e:
    print(f"Erro ao gerar Curva ROC: {e}")

print("\n\nTreinamento e Avaliação do Modelo Final CONCLUÍDOS.")