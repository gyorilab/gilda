import copy
import pandas
import requests
import itertools
from indra.databases import chebi_client

service_url = 'http://localhost:8001'

url = ('https://raw.githubusercontent.com/sorgerlab/famplex_paper/master/'
       'step4_stmt_entity_stats/test_agents_with_fplx_sample_curated.csv')

df = pandas.read_csv(url)

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
        'correct': bool(int(row['Grounding']))
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


# This dict contains counts of all the possible relationships between
# the reference grounding and the one produced by Gilda
comparison = {
    'ungrounded_ungrounded': 0,
    'ungrounded_grounded': 0,
    'ungrounded_correct': 0,
    'ungrounded_incorrect': 0,
    'incorrect_ungrounded': 0,
    'incorrect_not_matching': 0,
    'incorrect_matching': 0,
    'incorrect_correct': 0,
    'incorrect_incorrect': 0,
    'correct_ungrounded': 0,
    'correct_not_matching': 0,
    'correct_matching': 0,
    'correct_correct': 0,
    'correct_incorrect': 0
    }

incorrect_not_matching = []
correct_not_matching = []
correct_ungrounded = []
ungrounded_grounded = []
for grounding in groundings:
    # Send grounding requests
    res = requests.post('%s/ground' % service_url,
                        json={'text': grounding['text']})

    entry = None
    # If there is a grounding returned
    if res.json():
        # Get the top entry in the response which we consider the
        # grounding
        entry = res.json()[0]['term']
        # If the reference is ungrounded
        if not grounding['db_refs']:
            if grounding['text'] in correct_assertions:
                if entry['id'] == \
                        correct_assertions[grounding['text']].get(
                            entry['db']):
                    comparison['ungrounded_correct'] += 1
            elif grounding['text'] in incorrect_assertions:
                if entry['id'] == \
                        incorrect_assertions[grounding['text']].get(
                            entry['db']):
                    comparison['ungrounded_incorrect'] += 1
            else:
                ungrounded_grounded.append((grounding, entry))
                comparison['ungrounded_grounded'] += 1
        # If the reference is known to be incorrect
        elif not grounding['correct']:
            # If the grounding matches one of the known incorrect ones
            if grounding['db_refs'].get(entry['db']) == entry['id']:
                comparison['incorrect_matching'] += 1
            # Otherwise the grounding is not matching the known incorrect
            # one and we can't determine its correctness
            else:
                if grounding['text'] in correct_assertions:
                    if entry['id'] == \
                            correct_assertions[grounding['text']].get(
                                entry['db']):
                        comparison['incorrect_correct'] += 1
                elif grounding['text'] in incorrect_assertions:
                    if entry['id'] == \
                        incorrect_assertions[grounding['text']].get(
                            entry['db']):
                        comparison['incorrect_incorrect'] += 1
                else:
                    incorrect_not_matching.append((grounding, entry))
                    comparison['incorrect_not_matching'] += 1
        # If the reference is known to be correct
        else:
            # If the grounding matches one of the known correct ones
            if entry['id'] == grounding['db_refs'].get(entry['db']):
                comparison['correct_matching'] += 1

            elif grounding['text'] in correct_assertions:
                if entry['id'] == \
                        correct_assertions[grounding['text']].get(
                            entry['db']):
                    comparison['correct_correct'] += 1
            elif grounding['text'] in incorrect_assertions:
                if entry['id'] == \
                        incorrect_assertions[grounding['text']].get(
                            entry['db']):
                    comparison['correct_incorrect'] += 1
            # Otherwise we got a different grounding from the known correct
            # one and so we assume that its incorrect
            else:
                comparison['correct_not_matching'] += 1
                correct_not_matching.append((grounding, entry))
    # The remaining cases account for the case when we got no grounding
    # but the reference is either correct, incorrect, or ungrounded
    elif grounding['correct']:
        comparison['correct_ungrounded'] += 1
        correct_ungrounded.append((grounding, entry))
    elif not grounding['correct'] and not grounding['db_refs']:
        comparison['ungrounded_ungrounded'] += 1
    elif not grounding['correct'] and grounding['db_refs']:
        comparison['incorrect_ungrounded'] += 1


# In each of the following tuples, the first element is the worst case and
# the second element is the best case estimate based on information we have
correct = (comparison['correct_matching'] +
           comparison['incorrect_correct'] +
           comparison['ungrounded_correct'] +
           comparison['correct_correct'],

           comparison['correct_matching'] +
           comparison['correct_correct'] +
           comparison['incorrect_correct'] +
           comparison['ungrounded_correct'] +
           comparison['incorrect_not_matching'] +
           comparison['ungrounded_grounded'] +
           comparison['correct_not_matching'])
incorrect = (comparison['incorrect_matching'] +
             comparison['incorrect_incorrect'] +
             comparison['correct_not_matching'] +
             comparison['correct_incorrect'] +
             comparison['ungrounded_grounded'] +
             comparison['ungrounded_incorrect'] +
             comparison['incorrect_not_matching'],

             comparison['incorrect_matching'] +
             comparison['incorrect_incorrect'] +
             comparison['ungrounded_incorrect'] +
             comparison['correct_incorrect'])
ungrounded = (comparison['correct_ungrounded'] +
              comparison['incorrect_ungrounded'] +
              comparison['ungrounded_ungrounded'])

old_correct = (comparison['correct_not_matching'] +
               comparison['correct_matching'] +
               comparison['correct_ungrounded'] +
               comparison['correct_correct'] +
               comparison['correct_incorrect'])
old_incorrect = (comparison['incorrect_not_matching'] +
                 comparison['incorrect_matching'] +
                 comparison['incorrect_ungrounded'] +
                 comparison['incorrect_correct'] +
                 comparison['incorrect_incorrect'])
old_ungrounded = (comparison['ungrounded_ungrounded'] +
                  comparison['ungrounded_grounded'] +
                  comparison['ungrounded_correct'] +
                  comparison['ungrounded_incorrect'])

assert old_correct + old_incorrect + old_ungrounded == 300


prec = (correct[0] / (correct[0] + incorrect[0]),
        correct[1] / (correct[1] + incorrect[1]))
recall = (correct[0] / (correct[0] + ungrounded),
          correct[1] / (correct[1] + ungrounded))
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
