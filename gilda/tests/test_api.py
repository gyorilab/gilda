from gilda.tests import appreq
from gilda.api import *
from gilda.term import Term


def test_api_ground():
    scores = ground('kras')
    assert appreq(scores[0].score, 0.9845), scores
    scores = ground('ROS', 'reactive oxygen')
    assert scores[0].term.db == 'MESH', scores
    assert scores[0].url == 'https://identifiers.org/mesh:D017382'


def test_get_models():
    models = get_models()
    assert len(models) > 500
    assert 'STK1' in models


def test_get_names():
    names = get_names('HGNC', '6407')
    assert len(names) > 5, names
    assert 'K-Ras' in names


def test_organisms():
    # Default human gene match
    matches1 = ground('SMN1')
    assert len(matches1) == 1
    assert matches1[0].term.db == 'HGNC'
    assert matches1[0].term.id == '11117'
    # Prioritize human gene match
    matches2 = ground('SMN1', organisms=['9606', '10090'])
    assert len(matches2) == 1
    assert matches2[0].term.db == 'HGNC'
    assert matches2[0].term.id == '11117'
    # Prioritize mouse, SMN is grounded correctly
    matches3 = ground('SMN', organisms=['10090', '9606'])
    assert len(matches3) == 2, matches3
    assert matches3[0].term.db == 'UP'
    assert matches3[0].term.id == 'P63163'
    # Here we use SMN again but prioritize human and get three bad groundings
    matches4 = ground('SMN', organisms=['9606', '10090'])
    assert len(matches4) == 2, matches4
    assert all(m.term.organism == '9606' for m in matches4)
    # Finally we try grounding SMN1 with mouse prioritized, don't find a match
    # and end up with the human gene grounding
    matches5 = ground('TDRD16A', organisms=['10090', '9606'])
    assert len(matches5) == 1, matches5
    assert matches5[0].term.db == 'HGNC', matches5
    assert matches5[0].term.id == '11117', matches5


def test_make_grounder():
    grounder = make_grounder([
        Term('a', 'A', 'X', '1', 'A', 'name', 'test'),
        Term('b', 'B', 'X', '2', 'B', 'name', 'test')
    ])
    assert grounder.ground('a')
    assert not grounder.ground('x')
