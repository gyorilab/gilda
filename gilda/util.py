from copy import deepcopy

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


indra_namespace_reverse = {
    v: k for k, v in indra_namespace_mappings.items()
}


def apply_indra_ns(scored_match):
    scored_match_copy = deepcopy(scored_match)
    scored_match_copy.term.db = indra_namespace_mappings[scored_match.term.db]
    return scored_match_copy
