from gilda.tests import appreq
from gilda.api import *


def test_api_ground():
    scores = ground('kras')
    assert appreq(scores[0].score, 0.9845), scores
    scores = ground('ROS', 'reactive oxygen')
    assert scores[0].term.db == 'MESH'
    assert scores[0].url == 'https://identifiers.org/mesh:D017382'


def test_get_models():
    models = get_models()
    assert len(models) > 500
    assert 'STK1' in models


def test_get_names():
    names = get_names('HGNC', '6407')
    assert len(names) > 5, names
    assert 'K-Ras' in names


def test_api_use_indra_ns():
    matches = ground('mek')
    assert matches[0].term.db == 'fplx'
    matches = ground('mek', use_indra_ns=False)
    assert matches[0].term.db == 'fplx'
    matches = ground('mek', use_indra_ns=True)
    assert matches[0].term.db == 'FPLX'
