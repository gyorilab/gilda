import os
import time
import json
import pickle
from collections import Counter
from adeft.modeling.classify import AdeftClassifier
from gilda.grounder import Grounder
from gilda.resources import get_grounding_terms
from indra_db.util import get_db
from indra_db.util.content_scripts import get_text_content_from_text_refs
from indra.databases import mesh_client
from indra.literature import pubmed_client
from indra.literature.adeft_tools import universal_extract_text

grounding_terms_file = os.path.join(os.path.dirname(__file__), os.pardir,
                                    os.pardir, 'resources',
                                    'grounding_terms.tsv')

gr = Grounder(get_grounding_terms())


def get_ambiguities(hgnc_only=False, skip_assertions=True, skip_names=True):
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
        # If the entries all point to the same ID, we skip it
        if len(db_ids) <= 1:
            continue
        # If there is a name in statuses, we skip it because it's prioritized
        if skip_names and 'name' in statuses:
            continue
        # We skip assertions because they are prioritized anyway
        if skip_assertions and 'assertion' in statuses:
            continue
        # We can't get CHEBI PMIDs yet
        if 'CHEBI' in dbs:
            continue
        if 'adeft' in sources:
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
            print('Filtered out: %s' % str(terms))
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


def get_text_content(pmid):
    # print('Getting %s' % pmid)
    db = get_db('primary')
    content = get_text_content_from_text_refs(text_refs={'PMID': pmid},
                                              db=db)
    if content:
        text = universal_extract_text(content)
        return text
    return None


def get_papers(ambig_terms):
    term_pmids = {}
    pmid_counter = Counter()
    for term in ambig_terms:
        key = (term.db, term.id)
        if term.db == 'HGNC':
            gene = term.entry_name
            try:
                term_pmids[key] = pubmed_client.get_ids_for_gene(gene)
            except ValueError:
                print('Could not get PMIDs for gene: %s' % gene)
                term_pmids[key] = []
            pmid_counter.update(term_pmids[key])
            time.sleep(0.5)
        elif term.db == 'MESH':
            pmids = pubmed_client.get_ids_for_mesh(term.id, major_topic=False)
            if len(pmids > 1000):
                pmids = pubmed_client.get_ids_for_mesh(term.id,
                                                       major_topic=True)
            term_pmids[key] = pmids[:1000]
            pmid_counter.update(term_pmids[key])
            time.sleep(0.5)
        else:
            print('Unhandled ambiguous term: %s' % str(key))

    texts = []
    labels = []
    for key, pmids in term_pmids.items():
        print('Loading %d PMIDs for %s' % (len(pmids), str(key)))
        pmids = [p for p in pmids if pmid_counter[p] == 1]
        for pmid in pmids:
            txt = get_text_content(pmid)
            if txt:
                texts.append(txt)
                labels.append('%s:%s' % key)
    return texts, labels


def rank_ambiguities(ambigs, str_counts):
    sorted_ambigs = sorted(ambigs, key=lambda x: str_counts.get(x[0].text, 0),
                           reverse=True)
    return sorted_ambigs


if __name__ == '__main__':
    with open('raw_agent_text_count.json') as fh:
        str_counts = json.load(fh)
    ambigs = get_ambiguities()
    ambigs = filter_out_duplicates(ambigs)
    ambigs = filter_out_shared_prefix(ambigs)
    ambigs = filter_out_mesh_proteins(ambigs)
    ambigs = rank_ambiguities(ambigs, str_counts)
    import ipdb; ipdb.set_trace()
    # find_families(ambigs)
    param_grid = {'C': [10.0], 'max_features': [100, 1000],
                  'ngram_range': [(1, 2)]}
    print('Found a total of %d ambiguities.' % len(ambigs))
    for ambig in ambigs:
        print('Learning model for: %s which has %d occurrences'
              '\n=======' % (str(ambig), str_counts[ambig[0].text]))
        fname = 'models/%s.pkl' % ambig[0].text.replace('/', '_')
        if os.path.exists(fname):
            print('Model exists at %s, skipping' % fname)
            continue
        texts, labels = get_papers(ambig)
        label_counts = Counter(labels)
        if len(label_counts) < 2 or any([v <= 1 for v in
                                         label_counts.values()]):
            print('Could not get labels for more than one entry, skipping')
            continue
        if sum(label_counts.values()) <= 5:
            print('Got no more than 5 PMIDs overall, skipping')
            continue
        cl = AdeftClassifier([ambig[0].text], list(set(labels)))
        cl.cv(texts, labels, param_grid, cv=5)
        print(cl.stats)
        obj = {'cl': cl, 'ambig': ambig}
        with open(fname, 'wb') as fh:
            pickle.dump(obj, fh)
        print()
