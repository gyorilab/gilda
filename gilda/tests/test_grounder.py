import os
from gilda.grounder import Grounder
from . import appreq


fname = os.path.join(os.path.dirname(__file__), os.pardir, 'resources',
                     'grounding_terms.tsv')
gr = Grounder(fname)


def test_grounder():
    entries = gr.lookup('kras')
    statuses = [e.status for e in entries]
    assert 'name' in statuses
    for entry in entries:
        if entry.status == 'name':
            assert entry.id == '6407', entry

    scores = gr.ground('kras')
    assert len(scores) == 1, scores
    assert appreq(scores[0][1], 0.8536), scores
    scores = gr.ground('k-ras')
    assert len(scores) == 1, scores
    assert appreq(scores[0][1], 0.8546), scores
    scores = gr.ground('KRAS')
    assert len(scores) == 1, scores
    assert appreq(scores[0][1], 0.85542), scores
    scores = gr.ground('bRaf')
    assert len(scores) == 1, scores
    assert appreq(scores[0][1], 0.85466), scores


def test_grounder_bug():
    # Smoke test to make sure the 'NA' entry in grounding terms doesn't get
    # turned into a None
    gr.ground('Na')


def test_grounder_num_entries():
    entries = gr.lookup('NPM1')
    assert len(entries) == 1, entries
    entries = gr.lookup('H4')
    assert len(entries) == 3, entries


