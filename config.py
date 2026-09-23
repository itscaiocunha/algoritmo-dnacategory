"""Configuração central do projeto: caminhos e hiperparâmetros.

Todos os caminhos são resolvidos a partir da raiz do repositório, então o
pipeline funciona independentemente do diretório de onde é executado.
"""
from pathlib import Path

# --- Diretórios ---
RAIZ = Path(__file__).resolve().parent
DIR_DADOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "models"
DIR_FIGURAS = RAIZ / "reports" / "figures"

# --- Hiperparâmetros ---
KMER_SIZE = 6
N_CLUSTERS = 200            # K-Means do imputador
K_VIZINHOS = 5              # KNN do imputador
TARGET_FUNCTION = "protein binding"  # GO term usado como classe positiva
TEST_SIZE = 0.2
SVM_C = 1.0
RANDOM_STATE = 42

CHUNK_IMPUTACAO = 5_000
CHUNK_ACHATAMENTO = 100_000

# --- Dados brutos (exportados do Ensembl BioMart) ---
ARQ_SEQUENCIAS = DIR_DADOS / "genes_export.csv"
ARQ_FUNCOES = DIR_DADOS / "gene_funcao.txt"
ARQ_ORTOLOGOS = DIR_DADOS / "gene_ortologos.txt"

# --- Dados intermediários (gerados pelo pipeline) ---
ARQ_MESTRE = DIR_DADOS / "dataset_mestre.csv"
ARQ_REFERENCIA = DIR_DADOS / "referencia_limpa_com_clusters.csv"
ARQ_MESTRE_IMPUTADO = DIR_DADOS / "dataset_mestre_IMPUTADO_FINAL.csv"
ARQ_PROGRESSO_IMPUTACAO = DIR_DADOS / "imputacao_progresso.txt"
ARQ_ACHATADO = DIR_DADOS / "dataset_FINAL_ACHATADO.csv"

# --- Modelos ---
ARQ_VECTORIZER = DIR_MODELOS / f"count_vectorizer_k{KMER_SIZE}.joblib"
ARQ_KMEANS = DIR_MODELOS / f"kmeans_n{N_CLUSTERS}_k{KMER_SIZE}.joblib"

# --- Figuras ---
ARQ_MATRIZ_CONFUSAO = DIR_FIGURAS / "matriz_confusao_svm.png"
ARQ_CURVA_ROC = DIR_FIGURAS / "curva_roc_svm.png"
