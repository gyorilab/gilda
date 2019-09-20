import json
import time
import pickle
from collections import Counter
from adeft.modeling.classify import AdeftClassifier
from indra.util import batch_iter
from indra_db.util import get_db
from indra_db.util.content_scripts import get_text_content_from_text_refs
from indra.literature import pubmed_client
from indra.literature.adeft_tools import universal_extract_text
from find_ambiguities import *

db = get_db('primary')


def get_text_content(pmid):
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
            if len(pmids) > 1000:
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
    print('Training data size: %s' % str(Counter(labels)))
    return texts, labels


def rank_ambiguities(ambigs, str_counts):
    sorted_ambigs = sorted(ambigs, key=lambda x: str_counts.get(x[0].text, 0),
                           reverse=True)
    return sorted_ambigs


def learn_model(ambig_terms, params):
    texts, labels = get_papers(ambig_terms)
    label_counts = Counter(labels)
    if any([v <= 5 for v in label_counts.values()]):
        print('Could not get enough labels for at least one entry, skipping')
        return None
    cl = AdeftClassifier([ambig_terms[0].text], list(set(labels)))
    cl.cv(texts, labels, params, cv=5)
    print(cl.stats)
    return cl


if __name__ == '__main__':
    with open('raw_agent_text_count.json') as fh:
        str_counts = json.load(fh)
    ambigs = get_ambiguities()
    ambigs = filter_out_duplicates(ambigs)
    ambigs = filter_out_shared_prefix(ambigs)
    ambigs = filter_out_mesh_proteins(ambigs)
    ambigs = rank_ambiguities(ambigs, str_counts)
    param_grid = {'C': [10.0], 'max_features': [100, 1000],
                  'ngram_range': [(1, 2)]}
    print('Found a total of %d ambiguities.' % len(ambigs))

    models = {}
    for idx, ambig_terms_batch in enumerate(batch_iter(ambigs, 100)):
        pickle_name = 'gilda_ambiguities_hgnc_mesh_%d.pkl' % idx
        for ambig_terms in ambig_terms_batch:
            entity_text = ambig_terms[0].text
            print()
            if entity_text in models:
                print('Model for %s already exists' % entity_text)
                continue
            else:
                terms_str = '\n> ' + '\n> '.join(str(t) for t in ambig_terms)
                print('Learning model for: %s which has %d occurrences'
                  '\n=======' % (terms_str, str_counts[entity_text]))
            cl = learn_model(ambig_terms, param_grid)
            models[entity_text] = {'cl': cl, 'ambig': ambig_terms}
            with open(pickle_name, 'wb') as fh:
                pickle.dump(models, fh)
