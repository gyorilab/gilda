__version__ = '1.1.0'

import logging

from .api import get_grounder, get_models, get_names, ground, make_grounder, annotate
from .grounder import Grounder, ScoredMatch
from .pandas_utils import ground_df, ground_df_map
from .term import Term, dump_terms

__all__ = [
    'ground',
    'annotate',
    'get_models',
    'get_names',
    'get_grounder',
    'make_grounder',
    "dump_terms",
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
