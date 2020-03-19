import os
from collections import defaultdict
from gilda.generate_terms import generate_famplex_terms, generate_hgnc_terms, \
    generate_mesh_terms, generate_uniprot_terms, generate_chebi_terms, \
    generate_go_terms, filter_out_duplicates
from indra.databases import mesh_client


resources = os.path.join(os.path.dirname(__file__), os.path.pardir,
                         'gilda', 'resources')


def is_protein(mesh_id):
    mesh_protein = 'D000602'
    mesh_enzyme = 'D045762'
    for parent in [mesh_protein, mesh_enzyme]:
        if mesh_client.mesh_isa(mesh_id, parent):
            return True
    return False


def is_chemical(mesh_id):
    tn = mesh_client.get_mesh_tree_number(mesh_id)
    return tn.startswith('D')


def dump_mappings(mappings, fname):
    with open(fname, 'w') as fh:
        for mesh_id, maps in sorted(mappings.items(), key=lambda x: x[0]):
            if len(maps) > 1:
                print('Multiple mappings for %s: %s' % (mesh_id, str(maps)))
                continue
            me, te = list(maps.values())[0]
            fh.write('\t'.join([me.db, me.id, me.entry_name,
                                te.db, te.id, te.entry_name]) + '\n')


def get_ambigs_by_db(ambigs):
    ambigs_by_db = defaultdict(list)
    for term in ambigs:
        ambigs_by_db[term.db].append(term)
    return dict(ambigs_by_db)


def get_mesh_mappings(ambigs):
    mappings_by_mesh_id = defaultdict(dict)
    for text, ambig_terms in ambigs.items():
        ambigs_by_db = get_ambigs_by_db(ambig_terms)
        print('Considering %s' % text)
        for term in ambig_terms:
            print('%s:%s %s' % (term.db, term.id, term.entry_name))
        order = [('FPLX', is_protein),
                 ('HGNC', is_protein),
                 ('CHEBI', is_chemical),
                 ('GO', lambda x: True)]
        me = ambigs_by_db['MESH'][0]
        for ns, mesh_constraint in order:
            if len(ambigs_by_db.get(ns, [])) == 1 and mesh_constraint(me.id):
                mappings_by_mesh_id[me.id][(ambigs_by_db[ns][0].db,
                                            ambigs_by_db[ns][0].id)] = \
                        (me, ambigs_by_db[ns][0])
                print('Adding mapping for %s' % ns)
                break
        print('--------------')
    return mappings_by_mesh_id


def find_ambiguities(terms):
    ambig_entries = defaultdict(list)
    for term in terms:
        # We consider it an ambiguity if the same text entry appears
        # multiple times
        ambig_entries[term.text].append(term)
    # It's only an ambiguity if there are two entries at least
    ambig_entries = {k: v for k, v in ambig_entries.items() if len(v) >= 2}
    # We filter out any ambiguities that contain not exactly one MeSH term
    ambig_entries = {k: v for k, v in ambig_entries.items()
                     if len([e for e in v if e.db == 'MESH']) == 1}
    print('Found a total of %d relevant ambiguities' % len(ambig_entries))
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
    dump_mappings(mappings, os.path.join(resources, '_mesh_mappings.tsv'))
