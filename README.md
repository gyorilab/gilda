# Gilda: Grounding Integrating Learned Disambiguation
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Build](https://github.com/indralab/gilda/actions/workflows/tests.yml/badge.svg)](https://github.com/indralab/gilda/actions)
[![Documentation](https://readthedocs.org/projects/gilda/badge/?version=latest)](https://gilda.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/gilda.svg)](https://badge.fury.io/py/gilda)
[![DOI](https://img.shields.io/badge/DOI-10.1093/bioadv/vbac034-green.svg)](https://doi.org/10.1093/bioadv/vbac034)

Gilda is a Python package and REST service that grounds (i.e., finds
appropriate identifiers in namespaces for) named entities in biomedical text.

Gyori BM, Hoyt CT, Steppi A (2022). Gilda: biomedical entity text normalization with machine-learned disambiguation as a service. Bioinformatics Advances, 2022; vbac034 [https://doi.org/10.1093/bioadv/vbac034](https://doi.org/10.1093/bioadv/vbac034).

## Installation
Gilda is deployed as a web service at http://grounding.indra.bio/ (see
Usage instructions below), however, it can also be used locally as a Python
package.

The recommended method to install Gilda is through PyPI as
```bash
pip install gilda
```
Note that Gilda uses a single large resource file for grounding, which is
automatically downloaded into the `~/.data/gilda/<version>` folder during
runtime (see [pystow](https://github.com/cthoyt/pystow#%EF%B8%8F%EF%B8%8F-configuration) for options to
configure the location of this folder).

Given some additional dependencies, the grounding resource file can
also be regenerated locally by running `python -m gilda.generate_terms`.

## Documentation and notebooks
Documentation for Gilda is available [here](https://gilda.readthedocs.io).
We also provide several interactive Jupyter notebooks to help use and customize Gilda:
- [Gilda Introduction](https://github.com/indralab/gilda/blob/master/notebooks/gilda_introduction.ipynb) provides an interactive tutorial for using Gilda.
- [Custom Grounders](https://github.com/indralab/gilda/blob/master/notebooks/custom_grounders.ipynb) shows several examples of how Gilda can be instantiated with custom
grounding resources.
- [Model Training](https://github.com/indralab/gilda/blob/master/models/model_training.ipynb) provides interactive sample code for training
new disambiguation models.

## Usage
Gilda can either be used as a REST web service or used programmatically
via its Python API. An introduction Jupyter notebook for using Gilda
is available at
https://github.com/indralab/gilda/blob/master/notebooks/gilda_introduction.ipynb

### Use as a Python package
For using Gilda as a Python package, the documentation at
http://gilda.readthedocs.org provides detailed descriptions of each module of
Gilda and their usage. A basic usage example for named entity normalization (NEN),
or _grounding_ is as follows:

```python
import gilda
scored_matches = gilda.ground('ER', context='Calcium is released from the ER.')
```

Gilda also implements a simple dictionary-based named entity recognition (NER)
algorithm that can be used as follows:

```python
import gilda
results = gilda.annotate('Calcium is released from the ER.')
```

### Use as a web service

The REST service accepts POST requests with a JSON header on the `/ground`
endpoint. There is a public REST service running at http://grounding.indra.bio
but the service can also be run locally as

```bash
python -m gilda.app
```
which, by default, launches the server at `localhost:8001` (for local usage
replace the URL in the examples below with this address).

Below is an example request using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "kras"}' http://grounding.indra.bio/ground
```

The same request using Python's request package would be as follows:

```python
import requests
requests.post('http://grounding.indra.bio/ground', json={'text': 'kras'})
```

The web service also supports multiple inputs in a single request on the
`ground_multi` endpoint, for instance

```python
import requests
requests.post('http://grounding.indra.bio/ground_multi',
              json=[
                  {'text': 'braf'},
                  {'text': 'ER', 'context': 'endoplasmic reticulum (ER) is a cellular component'}
              ]
          )
```

## Resource usage
Gilda loads grounding terms into memory when first used. If memory usage
is an issue, the following options are recommended.

1. Run a single instance of Gilda as a local web service that one or more
other processes send requests to.

2. Create a custom Grounder instance that only loads a subset of terms
appropriate for a narrow use case.

3. Gilda also offers an optional sqlite back-end which significantly decreases
memory usage and results in minor drop in the number of strings grounder per
unit time. The sqlite back-end database can be built as follows with an
optional `[db_path]` argument, which if used, should use the .db extension. If
not specified, the .db file is generated in Gilda's default resource folder.

```bash
python -m gilda.resources.sqlite_adapter [db_path]
```

A Grounder instance can then be instantiated as follows:

```python
from gilda.grounder import Grounder
gr = Grounder(db_path)
matches = gr.ground('kras')
```

## Run web service with Docker

After cloning the repository locally, you can build and run a Docker image
of Gilda using the following commands:

```shell
$ docker build -t gilda:latest .
$ docker run -d -p 8001:8001 gilda:latest
```

Alternatively, you can use `docker-compose` to do both the initial build and
run the container based on the `docker-compose.yml` configuration:

```shell
$ docker-compose up
```

## Citation

```bibtex
@article{gyori2022gilda,
    author = {Gyori, Benjamin M and Hoyt, Charles Tapley and Steppi, Albert},
    title = "{{Gilda: biomedical entity text normalization with machine-learned disambiguation as a service}}",
    journal = {Bioinformatics Advances},
    year = {2022},
    month = {05},
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbac034},
    url = {https://doi.org/10.1093/bioadv/vbac034},
    note = {vbac034}
}
```

## Funding
The development of Gilda was funded under the DARPA Communicating with Computers
program (ARO grant W911NF-15-1-0544) and the DARPA Young Faculty Award
(ARO grant W911NF-20-1-0255).
