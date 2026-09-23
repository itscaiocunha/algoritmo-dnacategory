def limpar_id(id_sujo):
    """Normaliza um ID Ensembl para o formato 'ENSG...'.

    Remove sufixos após '|' ou espaço e a versão ('.N').
    Retorna None se o ID não for um gene Ensembl.

    >>> limpar_id('ENSG00000139618.15|BRCA2')
    'ENSG00000139618'
    """
    try:
        id_sem_pipe = str(id_sujo).split('|')[0].split()[0]
    except IndexError:  # string vazia ou só espaços
        return None
    id_sem_versao = id_sem_pipe.split('.')[0]
    if id_sem_versao.startswith('ENSG'):
        return id_sem_versao
    return None
