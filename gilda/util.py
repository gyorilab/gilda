indra_namespace_mappings = {
    'hgnc': 'HGNC',
    'uniprot': 'UP',
    'fplx': 'FPLX',
    'mesh': 'MESH',
    'go': 'GO',
    'chebi': 'CHEBI',
    'efo': 'EFO',
    'hp': 'HP',
    'doid': 'DOID',
}


def apply_indra_ns(scored_match):
    scored_match.term.db = indra_namespace_mappings[scored_match.term.db]
    return scored_match
