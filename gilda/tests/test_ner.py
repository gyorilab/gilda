from textwrap import dedent

import gilda
from gilda.ner import get_brat


def test_annotate():
    full_text = \
        "The protein BRAF is a kinase.\nBRAF is a gene.\nBRAF is a protein."

    annotations = gilda.annotate(full_text)
    assert isinstance(annotations, list)

    # Check that we get 7 annotations
    assert len(annotations) == 4

    # Check that the annotations are for the expected words
    assert tuple(a.text for a in annotations) == (
        'BRAF', 'kinase', 'BRAF', 'BRAF')

    # Check that the spans are correct
    expected_spans = ((12, 16), (22, 28), (30, 34), (46, 50))
    actual_spans = tuple((a.start, a.end) for a in annotations)
    assert actual_spans == expected_spans

    # Check that the curies are correct
    expected_curies = ("hgnc:1097", "mesh:D010770", "hgnc:1097", "hgnc:1097")
    actual_curies = tuple(a.matches[0].term.get_curie() for a in annotations)
    assert actual_curies == expected_curies


def test_get_brat():
    full_text = \
        "The protein BRAF is a kinase.\nBRAF is a gene.\nBRAF is a protein."

    brat_str = get_brat(gilda.annotate(full_text))

    assert isinstance(brat_str, str)
    match_str = dedent("""
        T1\tEntity 12 16\tBRAF
        #1\tAnnotatorNotes T1\thgnc:1097
        T2\tEntity 22 28\tkinase
        #2\tAnnotatorNotes T2\tmesh:D010770
        T3\tEntity 30 34\tBRAF
        #3\tAnnotatorNotes T3\thgnc:1097
        T4\tEntity 46 50\tBRAF
        #4\tAnnotatorNotes T4\thgnc:1097
        """).lstrip()
    assert brat_str == match_str


def test_get_all():
    full_text = "This is about ER."
    results = gilda.annotate(full_text)
    assert len(results) == 1
    curies = set()
    for annotation in results:
        for scored_match in annotation.matches:
            curies.add(scored_match.term.get_curie())
    assert "hgnc:3467" in curies  # ESR1
    assert "fplx:ESR" in curies
    assert "GO:0005783" in curies  # endoplasmic reticulum


def test_context_test():
    text = "This is about ER."
    context_text = "Estrogen receptor (ER) is a protein family."
    results = gilda.annotate(text, context_text=context_text)
    assert len(results) == 1
    assert results[0].matches[0].term.get_curie() == "fplx:ESR"
    assert results[0].text == "ER"
    assert (results[0].start, results[0].end) == (14, 16)

    context_text = "Calcium is released from the ER."
    results = gilda.annotate(text, context_text=context_text)
    assert len(results) == 1
    assert results[0].matches[0].term.get_curie() == "GO:0005783"
    assert results[0].text == "ER"
    assert (results[0].start, results[0].end) == (14, 16)


def test_punctuation_comma_in_entity():
    # A named entity with an actual comma in its name
    res = gilda.annotate('access, internet')
    assert len(res) == 1
    # Make sure we capture the text span exactly despite
    # tokenization
    assert res[0].text == 'access, internet'
    assert res[0].start == 0
    assert res[0].end == 16
    assert res[0].matches[0].term.db == 'MESH'
    assert res[0].matches[0].term.id == 'D000077230'


def test_punctuation_outside_entities():
    res = gilda.annotate('EGF binds EGFR, which is a receptor.')
    assert len(res) == 3

    assert [ann.text for ann in res] == ['EGF', 'EGFR', 'receptor']

    res = gilda.annotate('EGF binds EGFR: a receptor.')
    assert len(res) == 3

    assert [ann.text for ann in res] == ['EGF', 'EGFR', 'receptor']
