import pandas as pd
import sys

# --- Configuração ---
arquivo_csv = 'data/genes_imputados_knn_TESTE.csv'

# --- Função para Calcular Conteúdo GC ---
def calcular_conteudo_gc(sequencia):
    """Calcula a porcentagem de 'G' e 'C' em uma sequência de DNA."""
    try:
        if not isinstance(sequencia, str):
            return 0.0
        
        sequencia = sequencia.upper()
        total_bases = len(sequencia)
        if total_bases == 0:
            return 0.0
        
        # Conta Gs e Cs, ignorando Ns
        gc_count = sequencia.count('G') + sequencia.count('C')
        
        # Recalcula o total válido para GC (excluindo Ns)
        total_valido_gc = total_bases - sequencia.count('N')
        if total_valido_gc == 0:
            return 0.0
            
        gc_percent = (gc_count / total_valido_gc) * 100
        return gc_percent
        
    except Exception as e:
        print(f"Erro ao processar sequência: {e}")
        return 0.0

# --- 1. Carregar o Dataset ---
print(f"Carregando o dataset '{arquivo_csv}'...")
try:
    df = pd.read_csv(arquivo_csv)
except FileNotFoundError:
    print(f"ERRO: O arquivo '{arquivo_csv}' não foi encontrado.")
    sys.exit(1)
except Exception as e:
    print(f"ERRO ao ler o CSV: {e}")
    sys.exit(1)

print("Dataset carregado com sucesso.\n")

# --- 2. Inspeção Inicial ---
print("--- Inspeção Inicial dos Dados ---")
print(f"Total de registros de genes: {len(df)}")
print(f"Colunas encontradas: {list(df.columns)}\n")

# Garante que a coluna de sequência é do tipo string e preenche NaNs com string vazia para evitar erros
df['Sequencia'] = df['Sequencia'].astype(str).fillna('')

print(df.head())
print("-" * 30 + "\n")

# --- 3. Análise de Dados Faltantes ---
print("--- Análise de Dados Faltantes (Tipo 1: Células Vazias) ---")

# Conta quantas sequências são strings vazias
empty_seq_count = (df['Sequencia'] == '').sum()

print(f"Total de genes com sequência totalmente vazia: {empty_seq_count}")
if empty_seq_count > 0:
    print("IDs dos genes com sequências vazias:")
    print(df[df['Sequencia'] == ''][df.columns[0]].head())
print("-" * 30 + "\n")

# --- 4. Análise de Comprimento da Sequência ---
print("--- Análise de Comprimento da Sequência ---")

# Cria uma nova coluna com o comprimento de cada sequência
df['Comprimento'] = df['Sequencia'].apply(len)

# Filtra sequências vazias para estatísticas de comprimento
df_valido = df[df['Comprimento'] > 0]

if not df_valido.empty:
    comprimento_medio = df_valido['Comprimento'].mean()
    comprimento_mediano = df_valido['Comprimento'].median()
    comprimento_max = df_valido['Comprimento'].max()
    comprimento_min = df_valido['Comprimento'].min()

    print(f"Comprimento médio (genes > 0pb): {comprimento_medio:.2f} pb")
    print(f"Comprimento mediano (genes > 0pb): {comprimento_mediano:.0f} pb")
    print(f"Sequência mais longa: {comprimento_max} pb")
    print(f"Sequência mais curta: {comprimento_min} pb\n")

    gene_mais_longo = df_valido.loc[df_valido['Comprimento'].idxmax()]
    gene_mais_curto = df_valido.loc[df_valido['Comprimento'].idxmin()]

    print(f"Gene mais longo: {gene_mais_longo[df.columns[0]]} (Comprimento: {gene_mais_longo['Comprimento']})")
    print(f"Gene mais curto: {gene_mais_curto[df.columns[0]]} (Comprimento: {gene_mais_curto['Comprimento']})")
else:
    print("Nenhuma sequência válida encontrada para calcular estatísticas de comprimento.")
print("-" * 30 + "\n")

# --- 5. Análise de Dados Faltantes ---
print("--- Análise de Dados Faltantes (Tipo 2: Bases 'N') ---")

# Função para contar 'N's (bases desconhecidas)
def contar_bases_N(sequencia):
    return sequencia.upper().count('N')

# Cria uma coluna com a contagem de 'N's
df['N_Count'] = df['Sequencia'].apply(contar_bases_N)

# Calcula a porcentagem de 'N's em relação ao comprimento total
df['N_Percent_%'] = df.apply(
    lambda row: (row['N_Count'] / row['Comprimento']) * 100 if row['Comprimento'] > 0 else 0,
    axis=1
)

total_genes_com_N = (df['N_Count'] > 0).sum()
total_bases_N = df['N_Count'].sum()

print(f"Total de genes que contêm bases 'N': {total_genes_com_N}")
print(f"Total de bases 'N' (desconhecidas) no dataset: {total_bases_N}\n")

if total_genes_com_N > 0:
    print("Top 5 genes com maior PORCENTAGEM de 'N's:")
    df_sorted_n = df.sort_values(by='N_Percent_%', ascending=False)
    print(df_sorted_n[[df.columns[0], 'N_Percent_%', 'N_Count', 'Comprimento']].head())
print("-" * 30 + "\n")

# --- 6. Análise de Conteúdo GC ---
print("--- Análise de Conteúdo GC ---")

# Aplica a função para criar a coluna de % GC (função atualizada no início)
df['Conteudo_GC_%'] = df['Sequencia'].apply(calcular_conteudo_gc)

# Filtra para estatísticas de GC (onde o cálculo foi possível)
df_gc_valido = df[df['Conteudo_GC_%'] > 0]

if not df_gc_valido.empty:
    gc_medio = df_gc_valido['Conteudo_GC_%'].mean()
    gc_mediano = df_gc_valido['Conteudo_GC_%'].median()
    gc_max = df_gc_valido['Conteudo_GC_%'].max()
    gc_min = df_gc_valido['Conteudo_GC_%'].min()

    print(f"Conteúdo GC médio (em sequências válidas): {gc_medio:.2f}%")
    print(f"Conteúdo GC mediano (em sequências válidas): {gc_mediano:.2f}%")
    print(f"Maior conteúdo GC: {gc_max:.2f}%")
    print(f"Menor conteúdo GC: {gc_min:.2f}%\n")

    gene_gc_max = df_gc_valido.loc[df_gc_valido['Conteudo_GC_%'].idxmax()]
    gene_gc_min = df_gc_valido.loc[df_gc_valido['Conteudo_GC_%'].idxmin()]

    print(f"Gene com maior % GC: {gene_gc_max[df.columns[0]]} (GC: {gene_gc_max['Conteudo_GC_%']:.2f}%)")
    print(f"Gene com menor % GC: {gene_gc_min[df.columns[0]]} (GC: {gene_gc_min['Conteudo_GC_%']:.2f}%)")
else:
    print("Nenhuma sequência válida encontrada para calcular estatísticas de GC.")
print("-" * 30 + "\n")

# --- 7. Visualização Final dos Dados Processados ---
print("--- Amostra do DataFrame Final ---")
print(df.head())
print("\nAnálise concluída.")