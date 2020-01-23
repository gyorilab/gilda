import copy
import pandas
import requests
import itertools
from indra.databases import chebi_client

service_url = 'http://localhost:8001'

url = ('https://raw.githubusercontent.com/sorgerlab/famplex_paper/master/'
       'step4_stmt_entity_stats/test_agents_with_fplx_sample_curated.csv')

# Here we do some manual annotation to add expected correct groundings
# for some entries that were incorrect or ungrounded in the reference.
# This will allow us to compare more of the groundings to known correct
# groundings
correct_assertions = {'Stat': {'FPLX': 'STAT'},
                      'Thioredoxin Reductase': {'FPLX': 'TXNRD'},
                      'Thrombin': {'HGNC': '3535'},
                      'Aminopeptidases': {'MESH': 'D000626'},
                      'NF-AT proteins': {'MESH': 'D050778'},
                      'LTbetaR': {'HGNC': '6718'},
                      'RNAi': {'MESH': 'D034622'},
                      'Chaetocin': {'CHEBI': 'CHEBI:68747'},
                      'BAY11-7082': {'CHEBI': 'CHEBI:85928'},
                      'Toll-like receptors': {'MESH': 'D051193'},
                      'hippocalcin': {'HGNC': '5144'},
                      'alanine': {'CHEBI': 'CHEBI:16449'},
                      'H3K4me3': {'CHEBI': 'CHEBI:85043'},
                      'RXRalpha': {'HGNC': '10477'},
                      'chemokine receptors': {'MESH': 'D019707'},
                      'laminin': {'MESH': 'D007797'},
                      'Tiron': {'CHEBI': 'CHEBI:9607'},
                      'factor VII': {'MESH': 'D005167'},
                      'CD73': {'HGNC': '8021'},
                      'SERM': {'MESH': 'D020845'},
                      'angiotensin II': {'CHEBI': 'CHEBI:48432'},
                      'NAPA': {'HGNC': '7641'},
                      'IFN-beta': {'FPLX': 'IFNB'},
                      'cyclooxygenase-2': {'MESH': 'D051546'},
                      'Substance P': {'CHEBI': 'CHEBI:80308'},
                      'progestin': {'CHEBI': 'CHEBI:59826'}}


incorrect_assertions = {'IGF': {'HGNC': '5464'},
                        'DHM': {'CHEBI': 'CHEBI:71175'},
                        'SM': {'CHEBI': 'CHEBI:17076'},
                        'ARs': {'HGNC': '10015'}}


def process_fxpl_groundings(df):
    groundings = []
    # Iterate over the rows of the curation table and extract groundings
    for _, row in df.iterrows():
        if pandas.isnull(row['Grounding']):
            break
        # Here we get the original entity text, its type, and the
        # correct/incorrect curation
        grounding = {
            'text': row['Text'],
            'entity_type': row['EntityType'],
            'db_refs': {},
            'correct': bool(int(row['Grounding'])),
            'context': row['Sentence']
            }
        # We then extract the grounding (up to 3) that were considered
        # for the curation
        for i in [1, 2, 3]:
            if not pandas.isnull(row['DB_Ns%d' % i]):
                grounding['db_refs'][row['DB_Ns%d' % i]] = row['DB_Id%d' % i]
        # We standardize some of the grounding entries to match up with
        # Gilda's format
        for k, v in copy.deepcopy(grounding['db_refs']).items():
            # Strip off extra GO prefixes
            if v.startswith('GO:GO'):
                grounding['db_refs'][k] = v[3:]
            # Get CHEBI IDs from PUBCHEM
            if k == 'PUBCHEM':
                chebi_id = chebi_client.get_chebi_id_from_pubchem(v)
                if chebi_id:
                    grounding['db_refs']['CHEBI'] = 'CHEBI:%s' % chebi_id
        groundings.append(grounding)
    return groundings


def evaluate_old_grounding(grounding):
    """Return status of old grounding."""
    if not grounding['db_refs']:
        return 'ungrounded'
    elif not grounding['correct']:
        return 'incorrect'
    else:
        return 'correct'


def evaluate_new_grounding(grounding, term):
    """Return status of new grounding by comparing to old grounding."""
    if grounding['text'] in correct_assertions:
        if term['id'] == correct_assertions[grounding['text']].get(term['db']):
            return 'correct'
    elif grounding['text'] in incorrect_assertions:
        if term['id'] == \
                incorrect_assertions[grounding['text']].get(term['db']):
            return 'incorrect'
    elif not grounding['correct']:
        # If the grounding matches one of the known incorrect ones
        if grounding['db_refs'].get(term['db']) == term['id']:
            return 'incorrect'
    else:
        # If the grounding matches one of the known correct ones
        if grounding['db_refs'].get(term['db']) == term['id']:
            return 'correct'
    return 'unknown'


def make_comparison(groundings):
    # Generate an initial comparison matrix
    # This dict contains counts of all the possible relationships between
    # the reference grounding and the one produced by Gilda
    comparison = {'%s_%s' % (a, b): [] for a, b in
                  itertools.product(['ungrounded', 'correct', 'incorrect'],
                                    ['ungrounded', 'unknown', 'correct',
                                     'incorrect'])}

    # Now iterate over all the old groundings, get the new one, and build up
    # the values in the comparison matrix
    for idx, grounding in enumerate(groundings):
        old_eval = evaluate_old_grounding(grounding)
        # Send grounding requests
        res = requests.post('%s/ground' % service_url,
                            json={'text': grounding['text'],
                                  'context': grounding['context']}).json()
        if not res:
            comparison['%s_ungrounded' % old_eval].append((idx, grounding, None))
            continue
        term = res[0]['term']
        new_eval = evaluate_new_grounding(grounding, term)
        comparison['%s_%s' % (old_eval, new_eval)].append((idx, grounding, term))
    return comparison


def get_comparison_delta(groundings, c1, c2):
    def find_grounding(c, idx):
        for k, v in c.items():
            for entry in v:
                if entry[0] == idx:
                    return k, entry

    for idx, grounding in enumerate(groundings):
        c1g = find_grounding(c1, idx)
        c2g = find_grounding(c2, idx)
        if c1g[0] != c2g[0]:
            print(c1g, c2g)


# We now calculate various summary statistics and then print them
def get_sum_start(d, s):
    return sum([len(v) for k, v in d.items() if k.startswith(s)])


def get_sum_end(d, s):
    return sum([len(v) for k, v in d.items() if k.endswith(s)])


def print_statistics(comparison):
    old_correct = get_sum_start(comparison, 'correct')
    old_incorrect = get_sum_start(comparison, 'incorrect')
    old_ungrounded = get_sum_start(comparison, 'ungrounded')
    correct = get_sum_end(comparison, '_correct')
    incorrect = get_sum_end(comparison, '_incorrect')
    ungrounded = get_sum_end(comparison, '_ungrounded')
    unknown = get_sum_end(comparison, '_unknown')

    assert old_correct + old_incorrect + old_ungrounded == 300

    prec = (correct / (correct + incorrect + unknown),
            (correct + unknown) / (correct + incorrect + unknown))
    recall = (correct / (correct + ungrounded),
              (correct + unknown) / (correct + unknown + ungrounded))
    fscore = (2 * prec[0] * recall[0] / (prec[0] + recall[0]),
              2 * prec[1] * recall[1] / (prec[1] + recall[1]))

    old_prec = old_correct / (old_correct + old_incorrect)
    old_recall = old_correct / (old_correct + old_ungrounded)
    old_fscore = 2 * old_prec * old_recall / (old_prec + old_recall)

    print('The reference statistics were:')
    print('- Precision: %.3f\n- Recall: %.3f\n- F-score: %.3f' %
          (old_prec, old_recall, old_fscore))
    print('The current statistics with Gilda are between:')
    print('- Precision: (%.3f, %.3f)\n- Recall: (%.3f, %.3f)'
          '\n- F-score: (%.3f, %.3f)' %
          tuple(itertools.chain(prec, recall, fscore)))


if __name__ == '__main__':
    df = pandas.read_csv(url)
    groundings = process_fxpl_groundings(df)
    comparison = make_comparison(groundings)
    print_statistics(comparison)
