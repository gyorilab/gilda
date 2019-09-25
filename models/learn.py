import os
import json
import time
import pickle
import argparse
import functools
from multiprocessing import Pool
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


def get_pmids(ambig_terms):
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
    term_pmids = {k: [p for p in pmids if pmid_counter[p] == 1]
                  for k, pmids in term_pmids.items()}
    return term_pmids


def get_all_pmids(ambigs):
    all_pmids = [(a, get_pmids(a)) for a in ambigs]
    return all_pmids


def get_texts_for_term(key, pmids):
    texts = []
    print('Loading %d PMIDs for %s' % (len(pmids), str(key)))
    for pmid in pmids:
        txt = get_text_content(pmid)
        if txt:
            texts.append(txt)
    print('Loaded %d texts for %s' % (len(texts), str(key)))
    if texts and len(texts) <= 5:
        print('Splitting texts for %s' % str(key))
        texts = split_texts(texts, 6)
        print('Now have %s texts for %s' % (len(texts), str(key)))
    return texts


def get_papers(ambig_terms, term_pmids=None):
    if not term_pmids:
        term_pmids = get_pmids(ambig_terms)
    texts = []
    labels = []
    for key, pmids in term_pmids.items():
        texts1 = get_texts_for_term(key, pmids)
        texts += texts1
        labels += ['%s:%s' % key for _ in texts1]
    return texts, labels


def split_texts(texts, nmin):
    while len(texts) < nmin:
        texts = sorted(texts, key=lambda x: len(x), reverse=True)
        texts = [texts[0][:int(len(texts[0])/2)],
                 texts[0][int(len(texts[0])/2):]] + texts[1:]
    return texts


def rank_ambiguities(ambigs, str_counts):
    sorted_ambigs = sorted(ambigs, key=lambda x: str_counts.get(x[0].text, 0),
                           reverse=True)
    return sorted_ambigs


def learn_model(ambig_terms_pmids, params):
    ambig_terms, term_pmids = ambig_terms_pmids

    print()
    terms_str = '\n> ' + '\n> '.join(str(t) for t in ambig_terms)
    print('Learning model for: %s\n=======' % terms_str)
    texts, labels = get_papers(ambig_terms, term_pmids)
    if len(set(labels)) < 2:
        print('Could not get enough labels for more than one class, skipping.')
        return None

    label_counts = Counter(labels)
    if any([v <= 5 for v in label_counts.values()]):
        print('Could not get enough labels for at least one entry, skipping.')
        return None
    cl = AdeftClassifier([ambig_terms[0].text], list(set(labels)))
    cl.cv(texts, labels, params, cv=5)
    print(cl.stats)
    cl_model_info = cl.get_model_info()
    return {'cl': cl_model_info, 'ambig': ambig_terms}


def learn_batch(ambig_terms_batch):
    models = {}
    for ambig_terms in ambig_terms_batch:
        entity_text = ambig_terms[0].text
        res = learn_model(ambig_terms, param_grid)
        models[entity_text] = res
    return models


def get_filter_ambigs(str_counts):
    ambigs = get_ambiguities()
    ambigs = filter_out_duplicates(ambigs)
    ambigs = filter_out_shared_prefix(ambigs)
    ambigs = filter_out_mesh_proteins(ambigs)
    ambigs = rank_ambiguities(ambigs, str_counts)
    return ambigs


with open('raw_agent_text_count.json') as fh:
    str_counts = json.load(fh)


if __name__ == '__main__':
    # Initialization
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', default=1, type=int)
    args = parser.parse_args()
    pmid_cache_file = 'all_pmids.pkl'
    if os.path.exists(pmid_cache_file):
        # Cached PMIDs
        with open('all_pmids.pkl', 'rb') as fh:
            all_ambigs_pmids = pickle.load(fh)
    else:
        ambigs = get_filter_ambigs(str_counts)
        all_ambigs_pmids = get_all_pmids(ambigs)

    # Construct list of ambiguities
    #print('Found a total of %d ambiguities.' % len(ambigs))

    # Learn models
    param_grid = {'C': [10.0], 'max_features': [100, 1000],
                  'ngram_range': [(1, 2)]}
    models = {}
    if args.nproc == 1:
        for idx, ambig_terms_batch in \
                enumerate(batch_iter(all_ambigs_pmids, 10)):
            pickle_name = 'gilda_ambiguities_hgnc_mesh_%d.pkl' % idx
            models = learn_batch(ambig_terms_batch)
            with open(pickle_name, 'wb') as fh:
                pickle.dump(models, fh)
    else:
        pool = Pool(args.nproc)
        fun = functools.partial(learn_model, params=param_grid)
        pkl_idx = 0
        models = {}
        for count, model in enumerate(pool.imap_unordered(fun,
                                                          all_ambigs_pmids,
                                                          chunksize=10)):
            print('#### %d ####' % count)
            if model is None:
                print('Model is None, skipping')
            else:
                models[model['ambig'][0].text] = model
            if (count + 1) % 100 == 0:
                pickle_name = 'gilda_ambiguities_hgnc_mesh_%d.pkl' % pkl_idx
                with open(pickle_name, 'wb') as fh:
                    pickle.dump(models, fh)
                pkl_idx += 1
                models = {}
        pool.close()
        pool.join()

