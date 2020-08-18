from gilda.term import Term, get_identifiers_url


def test_standalone_get_url():
    assert get_identifiers_url('UP', 'P12345') == \
        'https://identifiers.org/uniprot:P12345'
    assert get_identifiers_url('HGNC', '12345') == \
        'https://identifiers.org/hgnc:12345'
    assert get_identifiers_url('CHEBI', 'CHEBI:12345') == \
        'https://identifiers.org/CHEBI:12345'


def test_term_get_url():
    term = Term(db='CHEBI', id='CHEBI:12345', entry_name='X',
                norm_text='x', text='X', source='test', status='name')
    assert term.get_idenfiers_url() == \
        'https://identifiers.org/CHEBI:12345'
