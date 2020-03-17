from gilda.tests import appreq
from gilda.api import *


def test_api_ground():
    scores = ground('kras')
    assert appreq(scores[0].score, 0.9845), scores
    scores = ground('ROS', 'reactive oxygen')
    assert scores[0].term.db == 'MESH'


def test_get_models():
    models = get_models()
    assert len(models) > 500
    assert 'STK1' in models


def test_get_names():
    names = get_names('HGNC', '6407')
    assert len(names) > 5, names
    assert 'K-Ras' in names
