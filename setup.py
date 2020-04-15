import re
from os import path
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as fh:
    long_description = fh.read()

with open(path.join(here, 'gilda', '__init__.py'), 'r') as fh:
    for line in fh.readlines():
        match = re.match(r'__version__ = \'(.+)\'', line)
        if match:
            gilda_version = match.groups()[0]
            break
    else:
        raise ValueError('Could not get version from gilda/__init__.py')



setup(name='gilda',
      version=gilda_version,
      description=('Grounding for biomedical entities with contextual '
                   'disambiguation'),
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/indralab/gilda',
      author='Benjamin M. Gyori, Harvard Medical School',
      author_email='benjamin_gyori@hms.harvard.edu',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
      packages=find_packages(),
      install_requires=['regex', 'adeft>=0.4.0', 'boto3', 'flask',
                        'flask-wtf', 'flask-bootstrap', 'obonet'],
      extras_require={'test': ['nose', 'coverage'],
                      'terms': ['indra'],
                      'benchmarks': ['pandas', 'requests']},
      keywords=['nlp', 'biology']
      )
