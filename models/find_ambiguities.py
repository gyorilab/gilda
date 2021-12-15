from gilda.grounder import Grounder
from indra.databases import mesh_client

gr = Grounder()


def get_ambiguities(skip_assertions=True, skip_names=True):
    ambig_entries = {}
    for terms in gr.entries.values():
        for term in terms:
            # We consider it an ambiguity if the same text entry appears
            # multiple times
            key = term.text
            if key in ambig_entries:
                ambig_entries[key].append(term)
            else:
                ambig_entries[key] = [term]
    # It's only an ambiguity if there are two entries at least
    ambig_entries = {k: v for k, v in ambig_entries.items() if len(v) >= 2}

    ambigs = []
    for text, entries in ambig_entries.items():
        dbs = {e.db for e in entries}
        db_ids = {(e.db, e.id) for e in entries}
        statuses = {e.status for e in entries}
        sources = {e.source for e in entries}
        names = {e.entry_name for e in entries}
        # If the entries all point to the same ID, we skip it
        if len(db_ids) <= 1:
            continue
        # If there is a name in statuses, we skip it because it's prioritized
        if skip_names and 'name' in statuses:
            continue
        # We skip curated terms because they are prioritized anyway
        if skip_assertions and 'curated' in statuses:
            continue
        # We can't get CHEBI PMIDs yet
        if 'CHEBI' in dbs:
            continue
        if 'adeft' in sources:
            continue
        # This typically happens for some UniProt IDs that don't have gene
        # names
        if any(n is None for n in names):
            continue
        # Everything else is an ambiguity
        ambigs.append(entries)
    return ambigs


def filter_out_shared_prefix(ambigs):
    non_long_prefix_ambigs = []
    for terms in ambigs:
        names = [term.entry_name for term in terms]
        prefix = lcp(names)
        if len(prefix) < 3:
            non_long_prefix_ambigs.append(terms)
    return non_long_prefix_ambigs


def filter_out_duplicates(ambigs):
    unique_ambigs = []
    for terms in ambigs:
        keys = set()
        unique_terms = []
        for term in terms:
            key = (term.db, term.id)
            if key not in keys:
                keys.add(key)
                unique_terms.append(term)
        unique_ambigs.append(unique_terms)
    return unique_ambigs


def filter_out_mesh_proteins(ambigs):
    mesh_protein = 'D000602'
    mesh_enzyme = 'D045762'
    all_new_ambigs = []
    for terms in ambigs:
        new_ambigs = []
        for term in terms:
            if term.db == 'MESH':
                if mesh_client.mesh_isa(term.id, mesh_protein) or \
                        mesh_client.mesh_isa(term.id, mesh_enzyme):
                    continue
            new_ambigs.append(term)
        if len(new_ambigs) >= 2:
            all_new_ambigs.append(new_ambigs)
        else:
            terms_str = '> ' + '\n> '.join(str(t) for t in terms)
            print('Filtered out:\n%s' % terms_str)
    return all_new_ambigs


def find_families(ambigs):
    # Find possible families
    for ambig in ambigs:
        names = [term.entry_name for term in ambig]
        prefix = lcp(names)
        if len(prefix) < 3:
            continue
        print('%s -> %s' % (ambig[0].text, ', '.join(sorted(names))))


def lcp(strs):
    # This function helps identify hypothetical families based on shared
    # prefixes
    prefix = ''
    for letters in zip(*strs):
        if len(set(letters)) == 1:
            prefix += letters[0]
        else:
            break
    return prefix

