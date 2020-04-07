import os
from collections import defaultdict
from gilda.generate_terms import *
from indra.databases import mesh_client


resources = os.path.join(os.path.dirname(__file__), os.path.pardir,
                         'gilda', 'resources')


def is_protein(mesh_id):
    return mesh_client.is_protein(mesh_id) or mesh_client.is_enzyme(mesh_id)


def is_chemical(mesh_id):
    return mesh_client.is_molecular(mesh_id)


def render_row(me, te):
    return '\t'.join([me.db, me.id, me.entry_name,
                      te.db, te.id, te.entry_name])


def get_nonambiguous(maps):
    # If there are more than one mappings from MESH
    if len(maps) > 1:
        # We see if there are any name-level matches
        name_matches = [(me, te) for me, te in maps
                        if me.entry_name.lower() == te.entry_name.lower()]
        # If we still have ambiguity, we print to the user
        if not name_matches or len(name_matches) > 1:
            print('Choose one if appropriate:')
            for me, te in maps:
                print(render_row(me, te))
            print('-----')
            return None
        # Otherwise. we add the single name matches mapping
        else:
            return name_matches[0]
    # If we map to only one thing, we keep that mapping
    else:
        return list(maps)[0]


def resolve_duplicates(mappings):
    keep_mappings = []
    # First we deal with mappings from MESH
    for maps in mappings.values():
        maps_list = maps.values()
        res = get_nonambiguous(maps_list)
        if res:
            keep_mappings.append(res)

    # Next we deal with mappings to MESH
    reverse_mappings = defaultdict(list)
    for mesh_term, other_term in keep_mappings:
        reverse_mappings[(other_term.db, other_term.id)].append((mesh_term,
                                                                 other_term))
    keep_mappings = []
    for maps in reverse_mappings.values():
        res = get_nonambiguous(maps)
        if res:
            keep_mappings.append(res)

    return keep_mappings


def dump_mappings(mappings, fname):
    with open(fname, 'w') as fh:
        for mesh_term, other_term in sorted(mappings, key=lambda x: x[0].id):
            fh.write(render_row(mesh_term, other_term) + '\n')


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
                 ('GO', lambda x: True),
                 ('DOID', lambda x: True),
                 ('HP', lambda x: True),
                 ('EFO', lambda x: True)]
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


def find_ambiguities(terms, match_attr='text'):
    match_fun = lambda x: x.text if match_attr == 'text' else x.norm_text
    ambig_entries = defaultdict(list)
    for term in terms:
        # We consider it an ambiguity if the same text entry appears
        # multiple times
        ambig_entries[match_fun(term)].append(term)
    # It's only an ambiguity if there are two entries at least
    ambig_entries = {k: v for k, v in ambig_entries.items() if len(v) >= 2}
    # We filter out any ambiguities that contain not exactly one MeSH term
    ambig_entries = {k: v for k, v in ambig_entries.items()
                     if len([e for e in v if e.db == 'MESH']) == 1}
    print('Found a total of %d relevant ambiguities' % len(ambig_entries))
    return ambig_entries


def get_terms():
    terms = generate_mesh_terms(ignore_mappings=True) + \
        generate_go_terms() + \
        generate_hgnc_terms() + \
        generate_famplex_terms() + \
        generate_uniprot_terms(download=False) + \
        generate_chebi_terms() + \
        generate_efo_terms() + \
        generate_hp_terms() + \
        generate_doid_terms()
    terms = filter_out_duplicates(terms)
    return terms


def manual_go_mappings(terms):
    td = defaultdict(list)
    for term in terms:
        td[(term.db, term.id)].append(term)
    # Migrated from FamPlex and INDRA
    map = [
        ('D002465', 'GO:0048870'),
        ('D002914', 'GO:0042627'),
        ('D012374', 'GO:0120200'),
        ('D014158', 'GO:0006351'),
        ('D014176', 'GO:0006412'),
        ('D018919', 'GO:0001525'),
        ('D048708', 'GO:0016049'),
        ('D058750', 'GO:0001837'),
        ('D059767', 'GO:0000725')
    ]
    mappings_by_mesh_id = defaultdict(dict)
    for mid, gid in map:
        mt = td[('MESH', mid)][0]
        gt = td[('GO', gid)][0]
        mappings_by_mesh_id[mid][('GO', gid)] = (mt, gt)
    return mappings_by_mesh_id


if __name__ == '__main__':
    terms = get_terms()
    # General ambiguities
    ambigs = find_ambiguities(terms, match_attr='text')
    mappings = get_mesh_mappings(ambigs)
    # Ambiguities that involve long strings but we allow normalized matches
    ambigs2 = find_ambiguities(terms, match_attr='norm_text')
    ambigs2 = {k: v for k, v in ambigs2.items() if len(k) > 6}
    mappings2 = get_mesh_mappings(ambigs2)
    for k, v in mappings2.items():
        if k not in mappings:
            mappings[k] = v
    mappings3 = manual_go_mappings(terms)
    for k, v in mappings3.items():
        if k not in mappings:
            mappings[k] = v
    mappings = resolve_duplicates(mappings)
    dump_mappings(mappings, os.path.join(resources, 'mesh_mappings.tsv'))
