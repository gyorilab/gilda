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
    assert tuple(a[0] for a in annotations) == (
        'protein', 'BRAF', 'kinase', 'BRAF', 'gene', 'BRAF', 'protein')

    # Check that the spans are correct
    assert annotations[0][2:4] == (4, 11)  # protein
    assert annotations[1][2:4] == (12, 16)  # BRAF
    assert annotations[2][2:4] == (22, 28)  # kinase
    assert annotations[3][2:4] == (30, 34)  # BRAF
    assert annotations[4][2:4] == (40, 44)  # gene
    assert annotations[5][2:4] == (46, 50)  # BRAF
    assert annotations[6][2:4] == (56, 63)  # protein

    # Check that the curies are correct
    assert isinstance(annotations[0][1], gilda.ScoredMatch)
    assert annotations[0][1].term.get_curie() == "CHEBI:36080"
    assert annotations[1][1].term.get_curie() == "hgnc:1097"
    assert annotations[2][1].term.get_curie() == "mesh:D010770"
    assert annotations[3][1].term.get_curie() == "hgnc:1097"
    assert annotations[4][1].term.get_curie() == "mesh:D005796"
    assert annotations[5][1].term.get_curie() == "hgnc:1097"
    assert annotations[6][1].term.get_curie() == "CHEBI:36080"


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
    results = gilda.annotate(full_text, return_first=False)
    assert len(results) > 1
    curies = {
        scored_match.term.get_curie()
        for _, scored_match, _, _ in results
    }
    assert "hgnc:3467" in curies  # ESR1
    assert "fplx:ESR" in curies
    assert "GO:0005783" in curies  # endoplasmic reticulum
