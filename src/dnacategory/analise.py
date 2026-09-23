"""Análise exploratória das sequências: comprimento, bases 'N' e conteúdo GC."""


def calcular_conteudo_gc(sequencia):
    """Porcentagem de G + C, ignorando bases 'N' no denominador."""
    if not isinstance(sequencia, str):
        return 0.0
    sequencia = sequencia.upper()
    total_valido = len(sequencia) - sequencia.count('N')
    if total_valido == 0:
        return 0.0
    return (sequencia.count('G') + sequencia.count('C')) / total_valido * 100


def contar_bases_n(sequencia):
    return sequencia.upper().count('N')


def adicionar_metricas(df):
    """Retorna uma cópia de df com colunas de comprimento, N e GC."""
    df = df.copy()
    df['Sequencia'] = df['Sequencia'].fillna('').astype(str)
    df['Comprimento'] = df['Sequencia'].str.len()
    df['N_Count'] = df['Sequencia'].apply(contar_bases_n)
    df['N_Percent_%'] = (df['N_Count'] / df['Comprimento'].where(df['Comprimento'] > 0) * 100).fillna(0)
    df['Conteudo_GC_%'] = df['Sequencia'].apply(calcular_conteudo_gc)
    return df


def _resumo(serie, unidade):
    return (f"média {serie.mean():.2f}{unidade} | mediana {serie.median():.2f}{unidade} | "
            f"min {serie.min():.2f}{unidade} | max {serie.max():.2f}{unidade}")


def relatorio(df):
    """Imprime um relatório exploratório e retorna o df com as métricas."""
    df = adicionar_metricas(df)
    coluna_id = df.columns[0]

    print(f"Total de genes: {len(df)}")
    print(f"Colunas: {list(df.columns)}\n")

    vazias = df['Comprimento'] == 0
    print(f"Genes com sequência vazia: {vazias.sum()}")

    validos = df[~vazias]
    if not validos.empty:
        print(f"Comprimento (pb): {_resumo(validos['Comprimento'], '')}")
        mais_longo = validos.loc[validos['Comprimento'].idxmax()]
        mais_curto = validos.loc[validos['Comprimento'].idxmin()]
        print(f"Mais longo: {mais_longo[coluna_id]} ({mais_longo['Comprimento']} pb)")
        print(f"Mais curto: {mais_curto[coluna_id]} ({mais_curto['Comprimento']} pb)\n")

    com_n = df['N_Count'] > 0
    print(f"Genes com bases 'N': {com_n.sum()}")
    print(f"Total de bases 'N': {df['N_Count'].sum()}")
    if com_n.any():
        print("Top 5 genes por % de 'N':")
        colunas = [coluna_id, 'N_Percent_%', 'N_Count', 'Comprimento']
        print(df.sort_values('N_Percent_%', ascending=False)[colunas].head(), "\n")

    gc = df[df['Conteudo_GC_%'] > 0]
    if not gc.empty:
        print(f"Conteúdo GC: {_resumo(gc['Conteudo_GC_%'], '%')}")

    return df
