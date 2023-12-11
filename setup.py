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
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
      ],
      packages=find_packages(),
      install_requires=['regex', 'adeft>=0.11.0', 'boto3', 'flask>=3.0,<4.0',
                        'flask-restx>=1.3.0', 'pystow>=0.1.10', 'unidecode',
                        'importlib_metadata; python_version < "3.8"',
                        'werkzeug'],
      extras_require={'test': ['pytest', 'pytest-cov', 'pandas'],
                      'terms': ['indra', 'obonet'],
                      'benchmarks': ['pandas>=1.0', 'requests',
                                     'tabulate', 'tqdm', 'click'],
                      'ui': [
                        'flask-wtf',
                        'flask-bootstrap',
                      ],
                      'docs': [
                          "sphinx",
                          "sphinx_autodoc_typehints",
                          "sphinx_rtd_theme",
                      ],
      },
      keywords=['nlp', 'biology'],
      include_package_data=True,
      entry_points={
          'console_scripts': [
              'gilda = gilda.app:main',
          ],
      },
      )
