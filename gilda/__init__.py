__version__ = '0.10.1'

import logging

from .api import get_grounder, get_models, get_names, ground, make_grounder
from .term import Term

__all__ = [
    'ground',
    'get_models',
    'get_names',
    'get_grounder',
    'make_grounder',
    # Classes
    'Term',
    # Meta
    '__version__',
]

logging.basicConfig(format=('%(levelname)s: [%(asctime)s] %(name)s'
                            ' - %(message)s'),
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('gilda')
