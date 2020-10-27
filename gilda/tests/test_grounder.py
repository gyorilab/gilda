import re
import logging
import requests
from gilda.grounder import Grounder
from . import appreq


logger = logging.getLogger(__name__)


gr = Grounder()


def test_validate_entries():
    logger.info('Getting identifiers registry...')
    url = ('https://registry.api.identifiers.org/resolutionApi/'
           'getResolverDataset')
    res = requests.get(url)
    identifiers_registry = {
        entry['prefix']: entry for entry in
        res.json()['payload']['namespaces']
    }
    logger.info('Got %d entries from identifiers registry...' %
                len(identifiers_registry))
    for terms in gr.entries.values():
        for term in terms:
            assert term.db in identifiers_registry,\
                '%s is not a valida namespace' % term.db
            assert re.match(identifiers_registry[term.db]['pattern'], term.id),\
                '%s is not a valid entry in the %s namespace' % \
                (term.id, term.db)


def test_grounder():
    entries = gr.lookup('kras')
    statuses = [e.status for e in entries]
    assert 'assertion' in statuses
    for entry in entries:
        if entry.status == 'assertion':
            assert entry.id == '6407', entry

    scores = gr.ground('kras')
    assert len(scores) == 1, scores
    assert appreq(scores[0].score, 0.9845), scores
    scores = gr.ground('k-ras')
    assert len(scores) == 1, scores
    assert appreq(scores[0].score, 0.9936), scores
    scores = gr.ground('KRAS')
    assert len(scores) == 1, scores
    assert appreq(scores[0].score, 1.0), scores
    scores = gr.ground('bRaf')
    assert len(scores) == 1, scores
    assert appreq(scores[0].score, 0.9936), scores


def test_grounder_bug():
    # Smoke test to make sure the 'NA' entry in grounding terms doesn't get
    # turned into a None
    gr.ground('Na')


def test_grounder_num_entries():
    entries = gr.lookup('NPM1')
    assert len(entries) == 1, entries
    entries = gr.lookup('H4')
    assert len(entries) == 5, entries


def test_grounder_depluralize():
    entries = gr.lookup('RAFs')
    assert len(entries) == 2, entries
    for entry in entries:
        assert entry.norm_text == 'raf'


def test_disambiguate_adeft():
    matches = gr.ground('IR')
    matches = gr.disambiguate('IR', matches, 'Insulin Receptor (IR)')
    for match in matches:
        assert match.disambiguation is not None
        assert match.disambiguation['type'] == 'adeft'
        assert match.disambiguation['match'] in ('grounded', 'ungrounded')
        assert match.disambiguation['score'] is not None
        if match.term.db == 'hgnc' and match.term.id == '6091':
            assert match.disambiguation['match'] == 'grounded', \
                match.disambiguation
            assert match.disambiguation['score'] == 1.0


def test_disambiguate_gilda():
    matches = gr.ground('NDR1')
    matches = gr.disambiguate('NDR1', matches, 'STK38')
    for match in matches:
        assert match.disambiguation['type'] == 'gilda'
        assert match.disambiguation['match'] == 'grounded'
        if match.term.db == 'hgnc' and match.term.id == '17847':
            assert match.disambiguation['score'] > 0.99
        if match.term.db == 'hgnc' and match.term.id == '7679':
            assert match.disambiguation['score'] < 0.01


def test_rank_namespace():
    matches = gr.ground('interferon-gamma')
    assert matches[0].term.db == 'hgnc', matches[0]


def test_aa_synonym():
    matches = gr.ground('WN')
    assert '141447' not in {m.term.id for m in matches}

    matches = gr.ground('W-N')
    assert '141447' not in {m.term.id for m in matches}


def test_use_indra_ns():
    matches = gr.ground('mek')
    assert matches[0].term.db == 'fplx'
    matches = gr.ground('mek', use_indra_ns=False)
    assert matches[0].term.db == 'fplx'
    matches = gr.ground('mek', use_indra_ns=True)
    assert matches[0].term.db == 'FPLX'
