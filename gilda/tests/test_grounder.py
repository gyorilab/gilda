from gilda.term import Term
from gilda.grounder import Grounder, filter_for_organism
from . import appreq


gr = Grounder()


def test_grounder():
    entries = gr.lookup('kras')
    statuses = [e.status for e in entries]
    assert 'curated' in statuses
    for entry in entries:
        if entry.status == 'curated':
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
    assert len(entries) == 4, entries
    entries = gr.lookup('H4')
    assert len(entries) == 7, entries


def test_grounder_depluralize():
    # Note that lookup returns all matches with no de-duplication
    # or filtering so we get two identical FPLX entries and a yeast protein
    # entry here.
    entries = gr.lookup('RAFs')
    assert len(entries) == 9, entries
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
        if match.term.db == 'HGNC' and match.term.id == '6091':
            assert match.disambiguation['match'] == 'grounded'
            assert match.disambiguation['score'] == 1.0


def test_disambiguate_gilda():
    matches = gr.ground('NDR1')
    matches = gr.disambiguate('NDR1', matches, 'STK38')
    for match in matches:
        assert match.disambiguation['type'] == 'gilda'
        assert match.disambiguation['match'] == 'grounded'
        if match.term.db == 'HGNC' and match.term.id == '17847':
            assert match.disambiguation['score'] > 0.99
        if match.term.db == 'HGNC' and match.term.id == '7679':
            assert match.disambiguation['score'] < 0.01


def test_rank_namespace():
    matches = gr.ground('interferon-gamma')
    assert matches[0].term.db == 'HGNC'


def test_aa_synonym():
    matches = gr.ground('WN')
    assert '141447' not in {m.term.id for m in matches}

    matches = gr.ground('W-N')
    assert '141447' not in {m.term.id for m in matches}


def test_organism_filter():
    dummy = 'dummy'
    t1 = Term('x', dummy, dummy, dummy, dummy, dummy, dummy, '9606')
    t2 = Term('x', dummy, dummy, dummy, dummy, dummy, dummy, '10090')
    t3 = Term('x', dummy, dummy, dummy, dummy, dummy, dummy, None)
    terms = filter_for_organism([t1, t2, t3],
                                organisms=['9606'])
    assert len(terms) == 2, terms
    assert {t.organism for t in terms} == {None, '9606'}
    terms = filter_for_organism([t1, t2, t3],
                                organisms=['10090'])
    assert len(terms) == 2, terms
    assert {t.organism for t in terms} == {None, '10090'}
    terms = filter_for_organism([t1, t2, t3],
                                organisms=['10090', '9606'])
    assert len(terms) == 2, terms
    assert {t.organism for t in terms} == {None, '10090'}
    terms = filter_for_organism([t1, t2, t3],
                                organisms=['9606', '10090'])
    assert len(terms) == 2, terms
    assert {t.organism for t in terms} == {None, '9606'}


def test_organisms():
    matches = gr.ground('Raf1')
    assert len(matches) == 2, len(matches)
    organisms = {match.term.organism for match in matches}
    assert organisms == {'9606'}, matches

    matches = gr.ground('Raf1', organisms=['10090'])
    assert len(matches) == 1, len(matches)
    organisms = {match.term.organism for match in matches}
    assert organisms == {'10090'}

    matches = gr.ground('Raf1', organisms=['9606', '10090'])
    assert len(matches) == 2, len(matches)
    organisms = {match.term.organism for match in matches}
    assert organisms == {'9606'}, matches

    matches = gr.ground('Raf1', organisms=['10090', '9606'])
    assert len(matches) == 1, len(matches)
    organisms = {match.term.organism for match in matches}
    assert organisms == {'10090'}, matches


def test_nonhuman_gene_synonyms():
    matches = gr.ground('Tau', organisms=['10090'])
    assert matches[0].term.db == 'UP', matches
    assert matches[0].term.id == 'P10637', matches


def test_uniprot_gene_synonym():
    matches = gr.ground('MEKK2')
    assert matches[0].term.db == 'HGNC', matches
    assert matches[0].term.entry_name == 'MAP3K2'


def test_greek_to_spelled_out():
    matches = gr.ground('interferon-γ')
    assert matches
    assert matches[0].term.entry_name == 'IFNG'


def test_roman_arabic_ground():
    for term in ['neurexin II', 'neurexin 2', 'neurexin-2', 'neurexin-ii']:
        matches = gr.ground(term)
        assert any((m.term.db, m.term.id) == ('HGNC', '8009') for m in matches)


def test_ground_go_activity():
    matches = gr.ground('EGFR')
    assert 'GO' not in {m.term.db for m in matches}, matches


def test_unidecode():
    txt = 'Löfgren’s syndrome'
    matches = gr.ground(txt)
    assert matches[0].term.db, matches[0].term.id == ('EFO', '0009466')

    txts = ['Aymé-Gripp syndrome', 'Ayme-Gripp syndrome']
    for txt in txts:
        matches = gr.ground(txt)
        assert len(matches) == 2
        assert {m.term.db for m in matches} == {'EFO', 'DOID'}

    txts = ['Bi₇O₉I₃', 'Bi7O9I3']
    for txt in txts:
        matches = gr.ground(txt)
        assert len(matches) == 1
        assert (matches[0].term.db, matches[0].term.id) == \
            ('MESH', 'C000605741')


def test_subsumed_terms():
    txt = 'mitochondria'
    matches = gr.ground(txt)
    assert len(matches) == 1
    match = matches[0]
    assert match.term.db == 'GO'
    assert len(match.subsumed_terms) == 1
    assert match.subsumed_terms[0].db == 'GO', match.subsumed_terms[0]
    assert match.subsumed_terms[0].source_db == 'MESH', match.subsumed_terms[0]
