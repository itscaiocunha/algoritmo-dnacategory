import pandas as pd
import numpy as np
from collections import Counter
import sys
import random
import os

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    print("ERRO: A biblioteca 'scikit-learn' não foi encontrada.")
    print("Por favor, instale-a com: pip install scikit-learn")
    sys.exit(1)

ARQUIVO_ENTRADA = '../../data/genes_export.csv'

def distancia_hamming(s1, s2):
    """
    Calcula a Distância Hamming (quantas posições são diferentes).
    """
    if len(s1) != len(s2):
        return abs(len(s1) - len(s2)) + max(len(s1), len(s2)) // 2
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def voto_majoritario(vizinhos, pos_i):
    """
    Olha a posição 'pos_i' nos k-vizinhos e retorna a base mais comum.
    """
    bases = [seq[pos_i] for seq in vizinhos if len(seq) > pos_i and seq[pos_i] != 'N']
    if not bases:
        return 'N'
    vencedor = Counter(bases).most_common(1)[0][0]
    return vencedor

def rodar_validacao_baseline(k=5, test_size=0.2, percent_faltante=0.05, n_rows=1500):
    print("="*50)
    print("INICIANDO TAREFA: VALIDAÇÃO DO MODELO BASELINE (KNN Global)")
    print(f"Parâmetros: k={k}, n_rows={n_rows}, percent_faltante={percent_faltante*100}%")
    print("="*50)
    
    print(f"Carregando dados de {ARQUIVO_ENTRADA} (Apenas {n_rows} linhas)...")
    try:
        df = pd.read_csv(ARQUIVO_ENTRADA, nrows=n_rows)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
        print(f"Verifique se o caminho está correto (esperando: {os.path.abspath(ARQUIVO_ENTRADA)})")
        return False

    df['Sequencia'] = df['Sequencia'].astype(str).fillna('')
    df_limpo = df[~df['Sequencia'].str.contains('N')].copy()

    if len(df_limpo) < 20:
        print(f"ERRO: Dados limpos insuficientes ({len(df_limpo)}) para validação (lido de {n_rows} linhas).")
        return False

    print(f"Total de {len(df_limpo)} sequências limpas encontradas (de {n_rows} lidas).")
    
    print(f"Dividindo dados: {1-test_size:.0%} para Referência, {test_size:.0%} para Validação.")
    df_referencia, df_teste_original = train_test_split(
        df_limpo, 
        test_size=test_size, 
        random_state=42
    )

    lista_referencia_seqs = df_referencia['Sequencia'].tolist()
    print(f"Total de vizinhos para referência: {len(lista_referencia_seqs)}")
    print(f"Total de sequências para teste (gabarito): {len(df_teste_original)}")

    y_true = []
    y_pred = []
    print(f"\nIniciando simulação... Serão 'furados' {percent_faltante:.0%} de cada gene de teste.")
    
    for index, linha in df_teste_original.iterrows():
        seq_original = linha['Sequencia']
        seq_id = linha.iloc[0]
        
        n_holes = int(len(seq_original) * percent_faltante)
        if n_holes == 0: continue

        try:
            indices_poked = random.sample(range(len(seq_original)), n_holes)
        except ValueError:
            continue
            
        seq_poked_list = list(seq_original)
        for pos in indices_poked:
            y_true.append(seq_original[pos])
            seq_poked_list[pos] = 'N'
        seq_poked_str = "".join(seq_poked_list)

        distancias = []
        for ref_seq in lista_referencia_seqs:
            dist = distancia_hamming(seq_poked_str, ref_seq)
            distancias.append(dist)
        
        indices_vizinhos = np.argsort(distancias)[:k]
        vizinhos_seqs = [lista_referencia_seqs[i] for i in indices_vizinhos]

        for pos in indices_poked:
            base_imputada = voto_majoritario(vizinhos_seqs, pos)
            y_pred.append(base_imputada)

    # Calcular Métricas
    print("\n--- Relatório de Desempenho (Baseline) ---")
    if not y_true:
        print("Nenhuma base foi imputada.")
        return False

    labels = sorted(list(set(y_true)))
    acc = accuracy_score(y_true, y_pred)
    print(f"Acurácia Geral: {acc * 100:.2f}%")
    print(f"(De {len(y_true)} bases 'N' simuladas, o KNN acertou a base correta em {acc*len(y_true):.0f} casos.)")
    print("\nRelatório de Classificação (Precisão, Recall, F1 por Base):")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    return True

if __name__ == "__main__":
    rodar_validacao_baseline(n_rows=1500)