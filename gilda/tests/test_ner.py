from textwrap import dedent

import gilda
from gilda.ner import get_brat


def test_annotate():
    full_text = \
        "The protein BRAF is a kinase.\nBRAF is a gene.\nBRAF is a protein."

    annotations = gilda.annotate(full_text)
    assert isinstance(annotations, list)

    # Check that we get 4 annotations
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
    assert "go:0005783" in curies  # endoplasmic reticulum


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
    assert results[0].matches[0].term.get_curie() == "go:0005783"
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


def test_synonym_corner_case():
    # Context: EFO contains the synonyms "transthyretin " with a trailing
    # space, this triggers an indexing error corner case when followed by -
    res = gilda.annotate('transthyretin -')
    assert res


def test_cell_death():
    anns = gilda.annotate("cell death")
    assert len(anns) == 1
    ann = anns[0]
    # Make sure that we match the entire span, including
    # the "cell" part that is in itself a stopword
    assert ann.text == "cell death", ann.text


def test_a_gene():
    # This tests that we don't ground spans that start with
    # a core stop word like a, the, etc.
    anns = gilda.annotate("a gene")
    assert not anns


def test_ui_example_annotation():
    text = (
        "Small G proteins are an extensive family of proteins that bind and "
        "hydrolyze GTP. They are ubiquitous inside cells, regulating a wide range "
        "of cellular processes. Recently, many studies have examined the role of "
        "small G proteins, particularly the Ras family of G proteins, in memory "
        "formation. Once thought to be primarily involved in the transduction of a "
        "variety of extracellular signals during development, it is now clear that "
        "Ras family proteins also play critical roles in molecular processing "
        "underlying neuronal and behavioral plasticity. We here review a number of "
        "recent studies that explore how the signaling of Ras family proteins "
        "contributes to memory formation. Understanding these signaling processes "
        "is of fundamental importance both from a basic scientific perspective, "
        "with the goal of providing mechanistic insights into a critical aspect of "
        "cognitive behavior, and from a clinical perspective, with the goal of "
        "providing effective therapies for a range of disorders involving cognitive "
        "impairments."
    )
    annotations = gilda.annotate(text)
    expected_annotations = [
        # Start, End, Text, Grounding
        (0, 16, "Small G proteins", "hgnc:9802"),
        (77, 80, "GTP", "chebi:15996"),
        (143, 161, "cellular processes", "go:0009987"),
        (212, 228, "small G proteins", "hgnc:9802"),
        (247, 250, "Ras", "fplx:RAS"),
        (261, 271, "G proteins", "fplx:G_protein"),
        (276, 282, "memory", "go:0007613"),
        (339, 351, "transduction", "go:0009293"),
        (368, 381, "extracellular", "go:0005576"),
        (431, 434, "Ras", "fplx:RAS"),
        (610, 619, "signaling", "go:0023052"),
        (623, 626, "Ras", "fplx:RAS"),
        (658, 664, "memory", "go:0007613"),
        (676, 689, "Understanding", "mesh:D032882"),
        (696, 715, "signaling processes", "go:0023052"),
        (871, 879, "behavior", "go:0007610"),
        (951, 960, "therapies", "mesh:D013812"),
        (976, 985, "disorders", "mesh:D004194"),
        (996, 1017, "cognitive impairments", "mesh:D060825"),
    ]
    assert len(annotations) == len(expected_annotations)
    for ann, expected in zip(annotations, expected_annotations):
        exp_start, exp_end, exp_text, exp_grounding = expected
        assert ann.start == exp_start
        assert ann.end == exp_end
        assert ann.text == exp_text
        assert ann.matches[0].term.get_curie() == exp_grounding
