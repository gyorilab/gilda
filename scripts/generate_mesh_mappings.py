import os
from collections import defaultdict
from gilda.generate_terms import generate_famplex_terms, generate_hgnc_terms, \
    generate_mesh_terms, generate_uniprot_terms, generate_chebi_terms, \
    generate_go_terms, filter_out_duplicates
from indra.databases import mesh_client


mesh_protein = 'D000602'
mesh_enzyme = 'D045762'
resources = os.path.join(os.path.dirname(__file__), os.path.pardir,
                         'gilda', 'resources')


def dump_mappings(mappings, fname):
    mappings = sorted(mappings.values(), key=lambda x: x[0].id)
    with open(fname, 'w') as fh:
        for me, te in mappings:
            fh.write('\t'.join([me.db, me.id, me.entry_name,
                                te.db, te.id, te.entry_name]) + '\n')


def get_ambigs_by_db(ambigs):
    ambigs_by_db = defaultdict(list)
    for term in ambigs:
        ambigs_by_db[term.db].append(term)
    return dict(ambigs_by_db)


def add_if_unique(mappings, ambigs_by_db, ns, mesh_constraints=None):
    me = ambigs_by_db['MESH'][0]
    if len(ambigs_by_db.get(ns, [])) == 1:
        if mesh_constraints:
            if not any(mesh_client.mesh_isa(me.id, mc)
                       for mc in mesh_constraints):
                return False
        key = (me.id, ns, ambigs_by_db[ns][0].id)
        mappings[key] = (me, ambigs_by_db[ns][0])
        return True
    return False


def get_mesh_mappings(ambigs):
    predicted_mappings = {}
    for text, ambig_terms in ambigs.items():
        ambigs_by_db = get_ambigs_by_db(ambig_terms)
        if len(ambigs_by_db.get('MESH', [])) != 1:
            continue
        print('Considering %s' % ambigs_by_db['MESH'][0].entry_name)
        order = [('FPLX', (mesh_protein, mesh_enzyme)),
                 ('HGNC', (mesh_protein, mesh_enzyme)),
                 ('CHEBI', None),
                 ('GO', None)]
        for ns, mesh_constraints in order:
            added = add_if_unique(predicted_mappings, ambigs_by_db, ns,
                                  mesh_constraints)
            if added:
                break
    return predicted_mappings


def find_ambiguities(terms):
    ambig_entries = defaultdict(list)
    for term in terms:
        # We consider it an ambiguity if the same text entry appears
        # multiple times
        ambig_entries[term.text].append(term)
    # It's only an ambiguity if there are two entries at least
    ambig_entries = {k: v for k, v in ambig_entries.items() if len(v) >= 2}
    return ambig_entries


def get_terms():
    terms = generate_mesh_terms(ignore_mappings=True) + \
        generate_hgnc_terms() + generate_famplex_terms() + \
        generate_uniprot_terms(download=False) + generate_chebi_terms() + \
        generate_go_terms()
    terms = filter_out_duplicates(terms)
    return terms


if __name__ == '__main__':
    terms = get_terms()
    ambigs = find_ambiguities(terms)
    mappings = get_mesh_mappings(ambigs)
    dump_mappings(mappings, os.path.join(resources, 'mesh_mappings.tsv'))
