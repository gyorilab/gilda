# Gilda: Grounding Integrating Learned Disambiguation
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Build](https://travis-ci.org/indralab/gilda.svg)](https://travis-ci.org/indralab/gilda)
[![Documentation](https://readthedocs.org/projects/gilda/badge/?version=latest)](https://gilda.readthedocs.io/en/latest/?badge=latest)

## Installation
Gilda is deployed as a web service at http://grounding.indra.bio/ground (see Usage
instructions below), it only needs to be installed if used locally.

The recommended method to install Gilda is via Github as:
```bash
pip install git+https://github.com/indralab/gilda.git
```
Note that Gilda uses a single large resource file for grounding, which is automatically downloaded
into the `~/.gilda/<version>` folder during runtime. Given some additional dependencies, the grounding
resource file can also be regenerated locally by running `python -m gilda.generate_terms`.

## Usage
Gilda can either be used as a REST service (recommended) or programmatically via its Python API.
An introduction Jupyter notebook for using Gilda as a service is available at
https://github.com/indralab/gilda/blob/master/notebooks/gilda_introduction.ipynb

The REST service accepts POST requests with a JSON header on the /ground endpoint.
There is a public REST service running on AWS but the service can also be run locally as

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

The above requests can also be used to interact with the public service, by using the
appropriate URL instead of `localhost:8001`.

As for using Gilda as a Python package, the documentation at
http://gilda.readthedocs.org provides detailed descriptions of each module
of Gilda and their usage.

## Funding
The development of Gilda is funded under the DARPA Communicating with Computers program (ARO grant W911NF-15-1-0544).
