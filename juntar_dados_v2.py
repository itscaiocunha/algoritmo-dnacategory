import pandas as pd
import numpy as np
import sys
import os

# --- 1. Configurações ---
# O "Godzilla" que o script 'aplicar_imputador.py' criou
ARQUIVO_ENTRADA = 'data/dataset_mestre_IMPUTADO_FINAL.csv' 
ARQUIVO_SAIDA = 'data/dataset_FINAL_ACHATADO.csv' # Nosso dataset de treino!

TARGET_FUNCTION = 'protein binding' # O alvo que queremos classificar
CHUNK_SIZE = 100000 # Ler 100k linhas do Godzilla de cada vez

# --- 2. Script Principal ---
print(f"Iniciando \"Achatamento\" RAM-Safe do dataset final...")
print(f"Arquivo de entrada: {ARQUIVO_ENTRADA} (~14.4M linhas)")
print(f"Arquivo de saída: {ARQUIVO_SAIDA} (~80k linhas)\n")

# Colunas que realmente importam
colunas_necessarias = ['join_key', 'Sequencia', 'GO term name']

# Listas para guardar os resultados "achatados" de cada chunk
lista_de_labels_flat = []
lista_de_seqs_flat = []

try:
    reader = pd.read_csv(ARQUIVO_ENTRADA, 
                         usecols=colunas_necessarias, 
                         chunksize=CHUNK_SIZE,
                         low_memory=False)
    
    for i, chunk in enumerate(reader):
        print(f"Processando Pedaço (Chunk) {i}...")
        
        # 1. Prepara o label (Y) no chunk
        # Se GO term for nulo (NaN), preenchemos com 'Sem_Funcao'
        chunk['GO term name'] = chunk['GO term name'].fillna('Sem_Funcao')
        chunk['label'] = np.where(chunk['GO term name'] == TARGET_FUNCTION, 1, 0)
        
        # 2. Achata o chunk (Y)
        # .max() garante que se *qualquer* anotação for 'protein binding', o gene é 1
        chunk_labels_flat = chunk.groupby('join_key')['label'].max()
        
        # 3. Achata o chunk (X)
        chunk_seq_flat = chunk.groupby('join_key')['Sequencia'].first()
        
        # 4. Guarda os resultados parciais
        lista_de_labels_flat.append(chunk_labels_flat)
        lista_de_seqs_flat.append(chunk_seq_flat)

    print("\nLeitura de chunks concluída. Consolidando dados...")

    # Consolida os resultados
    df_labels_parcial = pd.concat(lista_de_labels_flat)
    df_seqs_parcial = pd.concat(lista_de_seqs_flat)

    # Groupby FINAL
    print("Executando groupby final (Labels)...")
    # .max() de novo, para garantir que o '1' vença o '0' se um gene apareceu em chunks diferentes
    df_labels_flat = df_labels_parcial.groupby(level=0).max() 
    print("Executando groupby final (Sequências)...")
    df_seqs_flat = df_seqs_parcial.groupby(level=0).first()

    # Junta o X e Y finais
    df_final = pd.merge(df_seqs_flat, df_labels_flat, left_index=True, right_index=True, how='inner')
    
    print(f"\nDataset achatado com sucesso para {len(df_final)} genes únicos.")

except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
    sys.exit(1)
except MemoryError:
    print("\n--- ERRO DE MEMÓRIA (Mesmo com Chunks) ---")
    print("O Colab (versão gratuita) pode não ter RAM suficiente para consolidar os ~80k genes.")
    print("Tente reiniciar o ambiente e rodar apenas esta célula.")
    sys.exit(1)

# --- 4. Análise e Salvamento ---
print(f"Distribuição FINAL das classes:\n{df_final['label'].value_counts()}\n")

print(f"Salvando dataset de treino final em {ARQUIVO_SAIDA}...")
df_final.to_csv(ARQUIVO_SAIDA, index_label='join_key')

print("-" * 50)
print("PRONTO!")
print("Seu novo arquivo 'dataset_FINAL_ACHATADO.csv' contém TODOS os seus genes.")
print("Agora sim, podemos usar este arquivo para treinar o SVM/CNN.")