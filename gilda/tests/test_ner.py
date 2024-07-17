from textwrap import dedent

import gilda
from gilda.ner import get_brat


def test_annotate():
    full_text = \
        "The protein BRAF is a kinase.\nBRAF is a gene.\nBRAF is a protein."

    annotations = gilda.annotate(full_text)
    assert isinstance(annotations, list)

    # Check that we get 7 annotations
    assert len(annotations) == 7

    # Check that the annotations are for the expected words
    assert tuple(a.text for a in annotations) == (
        'protein', 'BRAF', 'kinase', 'BRAF', 'gene', 'BRAF', 'protein')

    # Check that the spans are correct
    expected_spans = ((4, 11), (12, 16), (22, 28), (30, 34), (40, 44),
                      (46, 50), (56, 63))
    actual_spans = tuple((a.start, a.end) for a in annotations)
    assert actual_spans == expected_spans

    # Check that the curies are correct
    expected_curies = ("CHEBI:36080", "hgnc:1097", "mesh:D010770",
                       "hgnc:1097", "mesh:D005796", "hgnc:1097",
                       "CHEBI:36080")
    actual_curies = tuple(a.matches[0].term.get_curie() for a in annotations)
    assert actual_curies == expected_curies


def test_get_brat():
    full_text = \
        "The protein BRAF is a kinase.\nBRAF is a gene.\nBRAF is a protein."

    brat_str = get_brat(gilda.annotate(full_text))

    assert isinstance(brat_str, str)
    match_str = dedent("""
        T1\tEntity 4 11\tprotein
        #1\tAnnotatorNotes T1\tCHEBI:36080
        T2\tEntity 12 16\tBRAF
        #2\tAnnotatorNotes T2\thgnc:1097
        T3\tEntity 22 28\tkinase
        #3\tAnnotatorNotes T3\tmesh:D010770
        T4\tEntity 30 34\tBRAF
        #4\tAnnotatorNotes T4\thgnc:1097
        T5\tEntity 40 44\tgene
        #5\tAnnotatorNotes T5\tmesh:D005796
        T6\tEntity 46 50\tBRAF
        #6\tAnnotatorNotes T6\thgnc:1097
        T7\tEntity 56 63\tprotein
        #7\tAnnotatorNotes T7\tCHEBI:36080
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
