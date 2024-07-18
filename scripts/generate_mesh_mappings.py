import os
import pandas
from collections import defaultdict
from gilda.generate_terms import *
from indra.databases import mesh_client


resources = os.path.join(os.path.dirname(__file__), os.path.pardir,
                         'gilda', 'resources')


def is_protein(mesh_id):
    return mesh_client.is_protein(mesh_id) or mesh_client.is_enzyme(mesh_id)


def is_chemical(mesh_id):
    return mesh_client.is_molecular(mesh_id)


def load_biomappings():
    """Load curated positive and negative mappings from Biomappings."""
    url_base = ('https://raw.githubusercontent.com/biopragmatics/biomappings/'
                'master/src/biomappings/resources/')
    positive_df = pandas.read_csv(url_base + 'mappings.tsv', sep='\t')
    negative_df = pandas.read_csv(url_base + 'incorrect.tsv', sep='\t')
    positive_mappings = defaultdict(list)
    negative_mappings = defaultdict(list)
    # These are the only relevant prefixes, there are mappings to
    # various other namespaces we don't need
    prefixes = {'fplx', 'chebi', 'go', 'hp', 'doid', 'efo', 'hgnc'}
    for mapping_df, mappings in ((positive_df, positive_mappings),
                                 (negative_df, negative_mappings)):
        for _, row in mapping_df.iterrows():
            # We only need exact matches.
            # TODO: should we consider non-exact matches to be effectively
            # negative?
            if row['relation'] != 'skos:exactMatch':
                continue
            # Look at both directions in which mesh mappings
            # can appear
            if row['source prefix'] == 'mesh':
                mesh_id = row['source identifier']
                other_ns = row['target prefix']
                other_id = row['target identifier']
            elif row['target prefix'] == 'mesh':
                mesh_id = row['target identifier']
                other_ns = row['source prefix']
                other_id = row['source identifier']
            else:
                continue
            if other_ns not in prefixes:
                continue
            # We make the namespace upper to be consistent
            # with Gilda
            mappings[mesh_id].append((other_ns.upper(), other_id))
    return positive_mappings, negative_mappings


def get_nonambiguous(maps):
    # If there is more than one mapping from MESH
    if len(maps) > 1:
        # We see if there are any name-level matches
        name_matches = [(me, te) for me, te in maps
                        if (me.entry_name.lower() if me.entry_name else '')
                            == (te.entry_name.lower() if te.entry_name else '')
                        # Corner case where we have multiple MeSH-based terms
                        # due to an orignal term from e.g., DOID having been
                        # mapped to MeSH
                        and me.db != te.db]

        # If we still have ambiguity, we print to the user
        if not name_matches or len(name_matches) > 1:
            return None, maps
        # Otherwise. we add the single name matches mapping
        else:
            return name_matches[0], []
    # If we map to only one thing, we keep that mapping
    else:
        return list(maps)[0], []


def resolve_duplicates(mappings):
    keep_mappings = []
    all_ambigs = []
    # First we deal with mappings from MESH
    for key, maps in mappings.items():
        maps_list = maps.values()
        keep, ambigs = get_nonambiguous(maps_list)
        if keep:
            keep_mappings.append(keep)
        if ambigs:
            all_ambigs += ambigs

    # Next we deal with mappings to MESH
    reverse_mappings = defaultdict(list)
    for mesh_term, other_term in keep_mappings:
        reverse_mappings[(other_term.db, other_term.id)].append((mesh_term,
                                                                 other_term))
    keep_mappings = []
    for maps in reverse_mappings.values():
        keep, ambigs = get_nonambiguous(maps)
        if keep:
            keep_mappings.append(keep)
        if ambigs:
            all_ambigs += ambigs

    return keep_mappings, all_ambigs


def dump_mappings(mappings, fname):
    def render_row(me, te):
        return '\t'.join([me.db, me.id, me.entry_name,
                          te.db, te.id, te.entry_name])

    with open(fname, 'w') as fh:
        for mesh_term, other_term in sorted(mappings, key=lambda x: x[0].id):
            # Corner case where we have multiple MeSH-based terms
            # due to an orignal term from e.g., DOID having been
            # mapped to MeSH
            if other_term.db != 'MESH':
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
        #print('Considering %s' % text)
        #for term in ambig_terms:
        #    print('%s:%s %s' % (term.db, term.id, term.entry_name))
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
                        [me, ambigs_by_db[ns][0]]
                #print('Adding mapping for %s' % ns)
                break
        #print('--------------')
    return dict(mappings_by_mesh_id)


def find_ambiguities(terms, match_attr='text'):
    match_fun = lambda x: x.text if match_attr == 'text' else x.norm_text
    ambig_entries = defaultdict(list)
    for term in terms:
        # We consider it an ambiguity if the same text entry appears
        # multiple times
        ambig_entries[match_fun(term)].append(term)
    # There is a corner case where the match_fun matches two different
    # synonyms / variants of the same entry from the same database which
    # are not really considered ambiguity but need to be reduced to a single
    # entry to avoid being inadvertently filtered out later
    ambig_entries = {
        # Here, we make sure we only keep a single term with a given db and id
        norm_term: list({(term.db, term.id): term for term in matching_terms}.values())
        for norm_term, matching_terms in ambig_entries.items()
    }
    # It's only an ambiguity if there are two entries at least
    ambig_entries = {norm_term: matching_terms
                     for norm_term, matching_terms
                     in ambig_entries.items() if len(matching_terms) >= 2}
    # We filter out any ambiguities that contain not exactly one MeSH term
    ambig_entries = {k: v for k, v in ambig_entries.items()
                     if len([e for e in v if e.db == 'MESH']) == 1}
    print('Found a total of %d relevant ambiguities' % len(ambig_entries))
    return dict(ambig_entries)


def get_terms():
    terms = generate_mesh_terms(ignore_mappings=True) + \
        generate_go_terms() + \
        generate_hgnc_terms() + \
        generate_famplex_terms(ignore_mappings=True) + \
        generate_uniprot_terms(download=False) + \
        generate_chebi_terms() + \
        generate_efo_terms(ignore_mappings=True) + \
        generate_hp_terms(ignore_mappings=True) + \
        generate_doid_terms(ignore_mappings=True)
    terms = filter_out_duplicates(terms)
    return terms


def manual_go_mappings(terms_by_id_tuple):
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
        mt = terms_by_id_tuple[('MESH', mid)]
        gt = terms_by_id_tuple[('GO', gid)]
        mappings_by_mesh_id[mid][('GO', gid)] = (mt, gt)
    return dict(mappings_by_mesh_id)


if __name__ == '__main__':
    terms = get_terms()
    # We create a lookup of term objects by their db/id tuple
    # for quick lookups. We also add source db/ids here
    # because they can be relevant when finding terms for
    # Biomappings curations. Note that when loading e.g.,
    # DOID terms, the native xrefs from DOID to MESH
    # are applied, even if terms are loaded with the ignore_mappings
    # option which just turns of loading the mappings that are
    # generated in this script.
    known_mappings = set()
    terms_by_id_tuple = {}
    for term in terms:
        terms_by_id_tuple[(term.db, term.id)] = term
        if term.source_id:
            terms_by_id_tuple[(term.source_db, term.source_id)] = term
            known_mappings.add((term.db, term.id, term.source_db, term.source_id))
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
    # Mappings from GO terms
    mappings3 = manual_go_mappings(terms_by_id_tuple)
    for k, v in mappings3.items():
        if k not in mappings:
            mappings[k] = v

    # We now have to account for Biomappings curations
    positive_biomappings, negative_biomappings = load_biomappings()
    keys_to_remove = set()
    # Iterate over all the automatically proposed mappings
    for mesh_id, local_mappings in mappings.items():
        # If we already have a positive curation for the given MeSH ID
        # we want to replace the content automatically generated here
        # with the terms corresponding to the positive curation
        if mesh_id in positive_biomappings:
            other_ids = positive_biomappings[mesh_id]
            new_mappings = {}
            for other_id in other_ids:
                # If the other ID already exists, we just copy it over
                if other_id in mappings[mesh_id]:
                    new_mappings[other_id] = mappings[mesh_id][other_id]
                # If it doesn't exist yet, we look up a Term for it
                # and add it to the mappings
                else:
                    if other_id in terms_by_id_tuple:
                        mesh_term = terms_by_id_tuple[('MESH', mesh_id)]
                        other_term = terms_by_id_tuple[other_id]
                        new_mappings[other_id] = [mesh_term, other_term]
                    # This is a corner case where something is in Biomappings
                    # but not in the set of Gilda terms. This can happen
                    # if a term has been deprecated/replaced in an ontology.
                    # We ignore these mappings and just keep what we have.
                    else:
                        print('%s missing from set of terms' % str(other_id))
                        new_mappings = mappings[mesh_id]
            mappings[mesh_id] = new_mappings
        # If we have a negative curation for this MeSH ID, we make sure
        # that we remove any known incorrect mappings
        if mesh_id in negative_biomappings:
            other_ids = negative_biomappings[mesh_id]
            if mesh_id in mappings:
                for other_id in other_ids:
                    if other_id in mappings[mesh_id]:
                        mappings[mesh_id].pop(other_id, None)
                # If nothing left, we remove the whole MeSH ID key
                if not mappings[mesh_id]:
                    keys_to_remove.add(mesh_id)
    for key in keys_to_remove:
        mappings.pop(key)
    nonambig_mappings, ambig_mappings = resolve_duplicates(mappings)
    dump_mappings(nonambig_mappings, os.path.join(resources, 'mesh_mappings.tsv'))
    dump_mappings(ambig_mappings,
                  os.path.join(resources, 'mesh_ambig_mappings.tsv'))

    # Known mappings are useful for debugging
    #with open(os.path.join(resources, 'known_mappings.tsv'), 'w') as fh:
    #    for db, id, source_db, source_id in sorted(known_mappings):
    #        fh.write('\t'.join([db, id, source_db, source_id]) + '\n')