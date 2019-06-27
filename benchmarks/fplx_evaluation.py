import copy
import pandas
import requests
import itertools
from indra.databases import chebi_client

url = ('https://raw.githubusercontent.com/sorgerlab/famplex_paper/master/'
       'step4_stmt_entity_stats/test_agents_with_fplx_sample_curated.csv')

df = pandas.read_csv(url)

groundings = []
for _, row in df.iterrows():
    if pandas.isnull(row['Grounding']):
        break
    grounding = {
        'text': row['Text'],
        'entity_type': row['EntityType'],
        'db_refs': {},
        'correct': bool(int(row['Grounding']))
        }
    for i in [1, 2, 3]:
        if not pandas.isnull(row['DB_Ns%d' % i]):
            grounding['db_refs'][row['DB_Ns%d' % i]] = row['DB_Id%d' % i]
    for k, v in copy.deepcopy(grounding['db_refs']).items():
        if v.startswith('GO:GO'):
            grounding['db_refs'][k] = v[3:]
        if k == 'PUBCHEM':
            chebi_id = chebi_client.get_chebi_id_from_pubchem(v)
            if chebi_id:
                grounding['db_refs']['CHEBI'] = 'CHEBI:%s' % chebi_id

    groundings.append(grounding)

comparison = {
    'ungrounded_ungrounded': 0,
    'incorrect_ungrounded': 0,
    'correct_ungrounded': 0,
    'ungrounded_grounded': 0,
    'incorrect_not_matching': 0,
    'incorrect_matching': 0,
    'correct_not_matching': 0,
    'correct_matching': 0
    }

incorrect_not_matching = []
correct_not_matching = []
correct_ungrounded = []
for grounding in groundings:
    res = requests.post('http://localhost:8001/ground',
                        json={'text': grounding['text']})
    entry = None
    if res.json():
        entry = res.json()[0]['term']
        if not grounding['db_refs']:
            comparison['ungrounded_grounded'] += 1
        elif not grounding['correct']:
            if entry['db'] in grounding['db_refs'] and \
                    entry['id'] == grounding['db_refs'][entry['db']]:
                comparison['incorrect_matching'] += 1
            else:
                incorrect_not_matching.append((grounding, entry))
                comparison['incorrect_not_matching'] += 1
        else:
            if entry['db'] in grounding['db_refs'] and \
                    entry['id'] == grounding['db_refs'][entry['db']]:
                comparison['correct_matching'] += 1
            else:
                comparison['correct_not_matching'] += 1
                correct_not_matching.append((grounding, entry))
    elif grounding['correct']:
        comparison['correct_ungrounded'] += 1
        correct_ungrounded.append((grounding, entry))
    elif not grounding['correct'] and not grounding['db_refs']:
        comparison['ungrounded_ungrounded'] += 1
    elif not grounding['correct'] and grounding['db_refs']:
        comparison['incorrect_ungrounded'] += 1


correct = (comparison['correct_matching'],
           comparison['correct_matching'] +
           comparison['incorrect_not_matching'] +
           comparison['ungrounded_grounded'])
incorrect = (comparison['incorrect_matching'] +
             comparison['correct_not_matching'] +
             comparison['ungrounded_grounded'],
             comparison['incorrect_matching'])
ungrounded = (comparison['correct_ungrounded'] +
              comparison['incorrect_ungrounded'] +
              comparison['ungrounded_ungrounded'])

old_correct = (comparison['correct_not_matching'] +
               comparison['correct_matching'] +
               comparison['correct_ungrounded'])
old_incorrect = (comparison['incorrect_not_matching'] +
                 comparison['incorrect_matching'] +
                 comparison['incorrect_ungrounded'])
old_ungrounded = (comparison['ungrounded_ungrounded'] +
                  comparison['ungrounded_grounded'])

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
