from gilda.term import Term, get_url


def test_standalone_get_url():
    assert get_url('UP', 'P12345') == 'https://bioregistry.io/uniprot:P12345'
    assert get_url('HGNC', '12345') == 'https://bioregistry.io/hgnc:12345'
    assert get_url('CHEBI', 'CHEBI:12345') == 'https://bioregistry.io/chebi:12345'


def test_term_get_url():
    term = Term(db='CHEBI', id='CHEBI:12345', entry_name='X',
                norm_text='x', text='X', source='test', status='name')
    assert term.get_curie() == 'chebi:12345'
    assert term.get_url() == 'https://bioregistry.io/chebi:12345'
    assert term.get_groundings() == {(term.db, term.id)}
    assert term.get_namespaces() == {term.db}


def test_term_source_db_id():
    term = Term('mitochondria', 'Mitochondria', 'GO', 'GO:0005739',
                'mitochondrion', 'synonym', 'mesh', None, 'MESH', 'D008928')
    assert term.source_db == 'MESH'
    assert term.source_id == 'D008928'
    assert term.get_groundings() == {(term.db, term.id),
                                     (term.source_db, term.source_id)}

    assert term.get_namespaces() == {term.db, term.source_db}
