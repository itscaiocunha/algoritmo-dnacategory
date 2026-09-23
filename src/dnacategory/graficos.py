"""Gráficos de avaliação do classificador."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # salva em arquivo, sem abrir janela

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, roc_curve  # noqa: E402


def plot_matriz_confusao(y_test, y_pred, path_saida, titulo):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Previsto: Outro (0)', 'Previsto: Alvo (1)'],
                yticklabels=['Real: Outro (0)', 'Real: Alvo (1)'])
    plt.title(titulo)
    plt.ylabel('Verdadeiro (Real)')
    plt.xlabel('Predito')
    _salvar(path_saida)


def plot_curva_roc(y_test, y_scores, path_saida, titulo):
    """Plota a curva ROC e retorna a AUC."""
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:0.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Acaso (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title(titulo)
    plt.legend(loc="lower right")
    _salvar(path_saida)
    return roc_auc


def _salvar(path_saida):
    Path(path_saida).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_saida)
    plt.close()
    print(f"Gráfico salvo em: {path_saida}")
