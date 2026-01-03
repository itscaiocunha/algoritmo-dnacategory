import pandas as pd
import sys
import os

# --- 1. Configuração de Caminhos ---
# (Ajuste os caminhos se estiverem diferentes)
PATH_SEQUENCIAS = 'data/genes_export.csv'
PATH_FUNCOES = 'data/gene_funcao.txt'
PATH_ORTOLOGOS = 'data/gene_ortologos.txt'
PATH_SAIDA = 'data/dataset_mestre.csv'

# --- 2. Função de Limpeza de ID ---
def limpar_id(id_sujo):
    """Limpa um ID de qualquer formato (com | ou .) para o formato ENSG..."""
    try:
        id_str = str(id_sujo)
        id_sem_pipe = id_str.split('|')[0].split()[0]
        id_sem_versao = id_sem_pipe.split('.')[0]
        if id_sem_versao.startswith('ENSG'):
            return id_sem_versao
        return None # Retorna None se não for um ID válido
    except:
        return None

# --- 3. Carregar e Processar Arquivo 1 (Sequências) ---
print(f"Carregando sequências de {PATH_SEQUENCIAS}...")
try:
    df_seq = pd.read_csv(PATH_SEQUENCIAS)
    # Pega a primeira coluna como ID, limpa, e define como a chave de junção
    df_seq['join_key'] = df_seq.iloc[:, 0].apply(limpar_id)
    df_seq = df_seq[['join_key', 'Sequencia']].dropna(subset=['join_key'])
    print(f"Encontradas {len(df_seq)} sequências com IDs limpos.")
except FileNotFoundError:
    print(f"ERRO: Arquivo de sequência '{PATH_SEQUENCIAS}' não encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"ERRO ao ler {PATH_SEQUENCIAS}: {e}")
    sys.exit(1)

# --- 4. Carregar e Processar Arquivo 2 (Funções/GO) ---
print(f"\nCarregando funções de {PATH_FUNCOES}...")
try:
    # O BioMart exporta como TSV (Separado por Tab), não CSV
    df_funcao = pd.read_csv(PATH_FUNCOES, sep=',')
    # Renomeia a coluna de ID e limpa
    df_funcao['join_key'] = df_funcao['Gene stable ID'].apply(limpar_id)
    # Mantém apenas as colunas úteis
    df_funcao = df_funcao[['join_key', 'GO term name', 'GO domain']].dropna(subset=['join_key'])
    print(f"Encontradas {len(df_funcao)} anotações de função (GO terms).")
except FileNotFoundError:
    print(f"ERRO: Arquivo de função '{PATH_FUNCOES}' não encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"ERRO ao ler {PATH_FUNCOES}: {e}")
    sys.exit(1)

# --- 5. Carregar e Processar Arquivo 3 (Ortólogos) ---
print(f"\nCarregando ortólogos de {PATH_ORTOLOGOS}...")
try:
    df_ortologos = pd.read_csv(PATH_ORTOLOGOS, sep=',')
    df_ortologos['join_key'] = df_ortologos['Gene stable ID'].apply(limpar_id)
    # Remove a coluna de ID original
    df_ortologos = df_ortologos.drop(columns=['Gene stable ID']).dropna(subset=['join_key'])
    print(f"Encontradas {len(df_ortologos)} anotações de ortólogos.")
except FileNotFoundError:
    print(f"ERRO: Arquivo de ortólogos '{PATH_ORTOLOGOS}' não encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"ERRO ao ler {PATH_ORTOLOGOS}: {e}")
    sys.exit(1)

# --- 6. Merge (Juntando tudo) ---
print("\nIniciando merge dos datasets...")

# Juntando Sequências + Funções
# (Note que isso vai duplicar linhas, pois um gene tem várias funções)
df_mestre = pd.merge(df_seq, df_funcao, on='join_key', how='left')

# Juntando o resultado com os Ortólogos
# (Precisamos remover duplicatas de 'join_key' nos ortólogos antes de juntar)
df_ortologos_unicos = df_ortologos.drop_duplicates(subset=['join_key'])
df_mestre = pd.merge(df_mestre, df_ortologos_unicos, on='join_key', how='left')

# Remove genes que não tiveram nenhuma função ou ortólogo encontrado
df_mestre = df_mestre.dropna(subset=['GO term name'])

print("Merge concluído.")

# --- 7. Salvar e Inspecionar ---
print(f"Salvando dataset mestre em {PATH_SAIDA}...")
os.makedirs(os.path.dirname(PATH_SAIDA), exist_ok=True)
df_mestre.to_csv(PATH_SAIDA, index=False)

print("\n--- Inspeção do Dataset Mestre ---")
print(f"Total de linhas (gene + anotação): {len(df_mestre)}")
print(f"Total de genes únicos: {df_mestre['join_key'].nunique()}")
print("\nColunas:")
print(df_mestre.info())
print("\nAmostra dos dados:")
print(df_mestre.head())

print("\n--- PRÓXIMO PASSO: ANÁLISE DE LABELS ---")
print("Como um gene tem várias funções, precisamos decidir qual será nosso 'Y'.")
print("Contagem das 10 principais 'Funções Moleculares':")
try:
    mf_counts = df_mestre[df_mestre['GO domain'] == 'molecular_function']['GO term name'].value_counts()
    print(mf_counts.head(10))
except KeyError:
    print("Coluna 'GO term name' ou 'GO domain' não encontrada. Verifique o .txt")

print("\nSUCESSO! Seu dataset mestre está pronto para a classificação.")