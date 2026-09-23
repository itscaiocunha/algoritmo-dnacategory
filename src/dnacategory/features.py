def get_kmers_string(sequence, k):
    """Converte uma sequência em k-mers separados por espaço.

    O formato "documento de palavras" permite usar o CountVectorizer do
    scikit-learn diretamente.

    >>> get_kmers_string('ATGCA', 3)
    'ATG TGC GCA'
    """
    seq_str = str(sequence)
    kmers = [seq_str[i:i + k] for i in range(len(seq_str) - k + 1)]
    return " ".join(kmers)


def sequencias_para_kmers(sequencias, k):
    """Aplica get_kmers_string a uma pd.Series de sequências."""
    return sequencias.astype(str).apply(lambda s: get_kmers_string(s, k))
