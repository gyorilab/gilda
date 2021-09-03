# Gilda: Grounding Integrating Learned Disambiguation
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Build](https://github.com/indralab/gilda/actions/workflows/tests.yml/badge.svg)](https://github.com/indralab/gilda/actions)
[![Documentation](https://readthedocs.org/projects/gilda/badge/?version=latest)](https://gilda.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/gilda.svg)](https://badge.fury.io/py/gilda)

Gilda is a Python package and REST service that grounds (i.e., finds
appropriate identifiers in namespaces for) named entities in biomedical text.

## Installation
Gilda is deployed as a web service at http://grounding.indra.bio/ (see
Usage instructions below), it only needs to be installed if used locally.

The recommended method to install Gilda is through PyPI as
```bash
pip install gilda
```
Note that Gilda uses a single large resource file for grounding, which is
automatically downloaded into the `~/.data/gilda/<version>` folder during
runtime (see [pystow](https://github.com/cthoyt/pystow#%EF%B8%8F-configuration) for options to
configure the location of this folder).

Given some additional dependencies, the grounding resource file can
also be regenerated locally by running `python -m gilda.generate_terms`.

## Usage
Gilda can either be used as a REST service or used programmatically
via its Python API. An introduction Jupyter notebook for using Gilda
is available at
https://github.com/indralab/gilda/blob/master/notebooks/gilda_introduction.ipynb

### Use via Python API
As for using Gilda as a Python package, the documentation at
http://gilda.readthedocs.org provides detailed descriptions of each module of
Gilda and their usage. A basic usage example is as follows

```python
import gilda
scored_matches = gilda.ground('ER', context='Calcium is released from the ER.')
```

### Use as a web service
The REST service accepts POST requests with a JSON header on the /ground
endpoint.  There is a public REST service running on AWS but the service can
also be run locally as

```bash
python -m gilda.app
```

Below is an example request using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "kras"}' http://localhost:8001/ground
```

The same request using Python's request package would be as follows:

```python
requests.post('http://localhost:8001/ground', json={'text': 'kras'})
```

The above requests can also be used to interact with the public service, by
using the appropriate URL instead of `localhost:8001`.

## Run web service with Docker

After cloning the repository locally, you can build and run using the
following two docker subcommands:

```shell
$ docker build -t gilda:latest .
$ docker run -d -p 8001:8001 gilda:latest
```

Alternatively, you can use `docker-compose` to do both the initial build and
run the container based on the `docker-compose.yml` configuration:

```shell
$ docker-compose up
```

## Funding
The development of Gilda is funded under the DARPA Communicating with Computers
program (ARO grant W911NF-15-1-0544).
