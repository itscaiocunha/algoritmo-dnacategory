"""Ponto de entrada do pipeline.

Uso:
    python pipeline.py juntar     # merge das fontes do BioMart -> dataset_mestre.csv
    python pipeline.py imputar    # treina (se preciso) e aplica o imputador de 'N'
    python pipeline.py achatar    # 1 linha por gene + rótulo binário
    python pipeline.py treinar    # SVM + métricas + gráficos
    python pipeline.py tudo       # todas as etapas acima, em ordem
    python pipeline.py analise    # relatório exploratório das sequências
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from dnacategory import analise, classificacao, dataset, graficos, imputacao  # noqa: E402


def etapa_juntar(args):
    df_mestre = dataset.juntar_dados(config.ARQ_SEQUENCIAS, config.ARQ_FUNCOES, config.ARQ_ORTOLOGOS)
    config.DIR_DADOS.mkdir(parents=True, exist_ok=True)
    df_mestre.to_csv(config.ARQ_MESTRE, index=False)
    print(f"Dataset mestre salvo em {config.ARQ_MESTRE}")
    print(f"\nTop funções moleculares:\n{dataset.top_funcoes_moleculares(df_mestre)}")


def etapa_imputar(args):
    modelos_existem = config.ARQ_VECTORIZER.exists() and config.ARQ_KMEANS.exists() \
        and config.ARQ_REFERENCIA.exists()

    if args.retreinar_imputador or not modelos_existem:
        print("Treinando imputador...")
        vectorizer, kmeans, df_ref = imputacao.treinar_imputador(
            pd.read_csv(config.ARQ_SEQUENCIAS), config.KMER_SIZE, config.N_CLUSTERS, config.RANDOM_STATE
        )
        imputacao.salvar_imputador(vectorizer, kmeans, df_ref,
                                   config.ARQ_VECTORIZER, config.ARQ_KMEANS, config.ARQ_REFERENCIA)
    else:
        print("Carregando imputador já treinado...")
        vectorizer, kmeans, df_ref = imputacao.carregar_imputador(
            config.ARQ_VECTORIZER, config.ARQ_KMEANS, config.ARQ_REFERENCIA
        )

    imputacao.aplicar_imputador(
        config.ARQ_MESTRE, config.ARQ_MESTRE_IMPUTADO, config.ARQ_PROGRESSO_IMPUTACAO,
        vectorizer, kmeans, df_ref, config.KMER_SIZE, config.K_VIZINHOS, config.CHUNK_IMPUTACAO,
    )


def etapa_achatar(args):
    df_final = dataset.achatar_dataset(config.ARQ_MESTRE_IMPUTADO, config.TARGET_FUNCTION,
                                       config.CHUNK_ACHATAMENTO)
    df_final.to_csv(config.ARQ_ACHATADO, index_label='join_key')
    print(f"Dataset de treino salvo em {config.ARQ_ACHATADO}")


def etapa_treinar(args):
    df_flat = pd.read_csv(config.ARQ_ACHATADO)
    resultado = classificacao.treinar_svm(df_flat, config.KMER_SIZE, config.TEST_SIZE,
                                          config.SVM_C, config.RANDOM_STATE)

    sufixo = f"SVM (k={config.KMER_SIZE}, Undersampled)"
    graficos.plot_matriz_confusao(resultado.y_test, resultado.y_pred,
                                  config.ARQ_MATRIZ_CONFUSAO, f"Matriz de Confusão - {sufixo}")
    roc_auc = graficos.plot_curva_roc(resultado.y_test, resultado.y_scores,
                                      config.ARQ_CURVA_ROC, f"Curva ROC - {sufixo}")
    print(f"AUC: {roc_auc:0.3f}")


def etapa_tudo(args):
    for etapa in (etapa_juntar, etapa_imputar, etapa_achatar, etapa_treinar):
        print(f"\n{'=' * 20} {etapa.__name__.removeprefix('etapa_').upper()} {'=' * 20}")
        etapa(args)


def etapa_analise(args):
    analise.relatorio(pd.read_csv(args.arquivo))


ETAPAS = {
    'juntar': etapa_juntar,
    'imputar': etapa_imputar,
    'achatar': etapa_achatar,
    'treinar': etapa_treinar,
    'tudo': etapa_tudo,
    'analise': etapa_analise,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('etapa', choices=ETAPAS)
    parser.add_argument('--retreinar-imputador', action='store_true',
                        help="força o treino do K-Means mesmo se já houver modelos salvos")
    parser.add_argument('--arquivo', type=Path, default=config.ARQ_SEQUENCIAS,
                        help="CSV analisado pela etapa 'analise' (padrão: genes_export.csv)")
    args = parser.parse_args()

    try:
        ETAPAS[args.etapa](args)
    except FileNotFoundError as e:
        print(f"ERRO: arquivo não encontrado: {e.filename}")
        print("Verifique se as etapas anteriores foram executadas (veja o README).")
        sys.exit(1)


if __name__ == "__main__":
    main()
