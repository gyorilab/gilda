import copy
import pandas
import itertools
from indra.databases import chebi_client
from gilda import ground
from tqdm import tqdm

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
                      'RNAi': {'MESH': 'D034622', 'GO': 'GO:0016441'},
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
                      'factor VII': {'HGNC': '3544'},
                      'CD73': {'HGNC': '8021'},
                      'SERM': {'MESH': 'D020845'},
                      'angiotensin II': {'CHEBI': 'CHEBI:48432'},
                      'NAPA': {'HGNC': '7641'},
                      'IFN-beta': {'FPLX': 'IFNB'},
                      'cyclooxygenase-2': {'HGNC': '9605'},
                      'substance P': {'CHEBI': 'CHEBI:80308'},
                      'progestin': {'CHEBI': 'CHEBI:59826'},
                      'MiR-125a': {'HGNC': '31505'},
                      'ROS': {'MESH': 'D017382'},
                      'PP5': {'HGNC': '9322'},
                      'aminopeptidases': {'MESH': 'D000626'},
                      'IMP1': {'HGNC': '28866'},
                      '293T': {'EFO': '0001082'},
                      'GR': {'HGNC': '7978'},
                      'integrin alpha': {'FPLX': 'ITGA'},
                      'DC': {'MESH': 'D003713'},
                      'BMD': {'MESH': 'D015519'},
                      'angina': {'MESH': 'D000787', 'EFO': '0003913'}}


incorrect_assertions = {'IGF': {'HGNC': '5464'},
                        'DHM': {'CHEBI': 'CHEBI:71175'},
                        'SM': {'CHEBI': 'CHEBI:17076'},
                        'BMD': {'HGNC': ['2928', '12703']},
                        'DC': {'HGNC': '2714'},
                        'ARs': {'HGNC': '644'},
                        'BA': {'CHEBI': 'CHEBI:32594'},
                        'HP1': {'HGNC': '1555'},
                        'PRF': {'UP': 'Q6P3D7'},
                        'CUL4': {'UP': 'Q17392'},
                        'MEKK3': {'GO': 'GO:0004709'},
                        'Hrs': {'GO': 'GO:0000725'},
                        'thioredoxin-1': {'UP': 'P47938'},
                        'alpha4': {'HGNC': '10809'},
                        'NT': {'HGNC': '17941'},
                        'IMP1': {'HGNC': '16435'}}


def process_fplx_groundings(df):
    groundings = []
    # Iterate over the rows of the curation table and extract groundings
    for _, row in tqdm(df.iterrows(), desc='Processing groundings'):
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
                    grounding['db_refs']['CHEBI'] = chebi_id
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
        if term.id == correct_assertions[grounding['text']].get(term.db):
            return 'correct'
    if grounding['text'] in incorrect_assertions:
        incorrect_val = incorrect_assertions[grounding['text']].get(term.db)
        if isinstance(incorrect_val, str):
            if term.id == incorrect_val:
                return 'incorrect'
        elif isinstance(incorrect_val, list):
            if term.id in incorrect_val:
                return 'incorrect'

    if not grounding['correct']:
        # If the grounding matches one of the known incorrect ones
        if grounding['db_refs'].get(term.db) == term.id:
            return 'incorrect'
    else:
        # If the grounding matches one of the known correct ones
        if grounding['db_refs'].get(term.db) == term.id:
            return 'correct'
    return 'unknown'


def make_comparison(groundings, use_disamb=True):
    # Generate an initial comparison matrix
    # This dict contains counts of all the possible relationships between
    # the reference grounding and the one produced by Gilda
    comparison = {'%s_%s' % (a, b): [] for a, b in
                  itertools.product(['ungrounded', 'correct', 'incorrect'],
                                    ['ungrounded', 'unknown', 'correct',
                                     'incorrect'])}

    # Now iterate over all the old groundings, get the new one, and build up
    # the values in the comparison matrix
    for idx, grounding in enumerate(tqdm(groundings, desc='Making comparison')):
        old_eval = evaluate_old_grounding(grounding)
        # Send grounding requests
        context = grounding['context'] if use_disamb else None
        matches = ground(text=grounding['text'],
                         context=context)
        if not matches:
            comparison['%s_ungrounded' % old_eval].append((idx, grounding,
                                                           None))
            continue
        term = matches[0].term
        new_eval = evaluate_new_grounding(grounding, term)
        comparison['%s_%s' % (old_eval, new_eval)].append((idx, grounding,
                                                           term))
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

    # Note: it can happen that a change in Gilda's behavior results in
    # producing a grounding that is not curated and is therefore of
    # "unknown" status. To cover this possibility, we calculate a lower
    # and an upper bound for these values
    prec = (correct / (correct + incorrect + unknown),
            (correct + unknown) / (correct + incorrect + unknown))
    recall = (correct / (correct + ungrounded),
              (correct + unknown) / (correct + unknown + ungrounded))
    fscore = (2 * prec[0] * recall[0] / (prec[0] + recall[0]),
              2 * prec[1] * recall[1] / (prec[1] + recall[1]))

    old_prec = old_correct / (old_correct + old_incorrect)
    old_recall = old_correct / (old_correct + old_ungrounded)
    old_fscore = 2 * old_prec * old_recall / (old_prec + old_recall)

    for k, v in comparison.items():
        print('%s: %s' % (k, len(v)))

    print('The reference statistics were:')
    print('- Precision: %.3f\n- Recall: %.3f\n- F-score: %.3f' %
          (old_prec, old_recall, old_fscore))
    print('The current statistics with Gilda are:')
    if unknown:
        print('- Precision: (%.3f, %.3f)\n- Recall: (%.3f, %.3f)'
              '\n- F-score: (%.3f, %.3f)' %
              tuple(itertools.chain(prec, recall, fscore)))
    else:
        print('- Precision: %.3f\n- Recall: %.3f\n- F-score: %.3f' %
              (prec[0], recall[0], fscore[0]))

    print()

    rows = [[old_prec, old_recall, old_fscore]]
    rows.extend(zip(prec, recall, fscore) if unknown \
                else [[prec[0], recall[0], fscore[0]]])
    df2 = pandas.DataFrame(
        rows,
        columns=["Precision", "Recall", "F_1"],
        index=["Reference", "Current_1", "Current_2"] if unknown else \
            ["Reference", "Current"],
    ).round(3)
    df2.columns.name = 'Trial'
    print(df2.to_latex(caption="FamPlex benchmarking results",
                       label='tab:famplex-benchmark-results'))

    df1 = pandas.DataFrame(
        [
            (*(x.capitalize() for x in k.split('_')), len(v))
            for k, v in comparison.items()
        ],
        columns=["Expected", "Actual", "Count"],
    )
    df1 = df1.pivot(index=["Expected"], columns=["Actual"], values=["Count"])
    print(df1.to_latex(caption="FamPlex benchmarking confusion matrix",
                       label="tab:famplex-confusion"))


def run_comparison(use_disamb=True):
    df = pandas.read_csv(url)
    groundings = process_fplx_groundings(df)
    comparison = make_comparison(groundings, use_disamb)
    print_statistics(comparison)
    return comparison


if __name__ == '__main__':
    comparison = run_comparison(use_disamb=True)
