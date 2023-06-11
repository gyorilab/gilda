__version__ = '0.12.0'

import logging

from .api import get_grounder, get_models, get_names, ground, make_grounder
from .grounder import Grounder, ScoredMatch
from .pandas_utils import ground_df, ground_df_map
from .term import Term

__all__ = [
    'ground',
    'get_models',
    'get_names',
    'get_grounder',
    'make_grounder',
    # Classes
    'Term',
    'Grounder',
    'ScoredMatch',
    # Meta
    '__version__',
    # Pandas utilities
    'ground_df',
    'ground_df_map',
]

logging.basicConfig(format=('%(levelname)s: [%(asctime)s] %(name)s'
                            ' - %(message)s'),
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('gilda')
