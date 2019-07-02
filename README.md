# Gilda: Grounding Integrating Learned Disambiguation
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
[![Build](https://travis-ci.org/bgyori/gilda.svg)](https://travis-ci.org/bgyori/gilda)
[![Documentation](https://readthedocs.org/projects/gilda/badge/?version=latest)](https://gilda.readthedocs.io/en/latest/?badge=latest)

## Installation
The recommended method to install Gilda is via Github as:
```bash
pip install git+https://github.com/bgyori/gilda.git
```
Note that Gilda uses a single large resource file for grounding, which is automatically downloaded
into the `~/.gilda/<version>` folder during runtime. Given some additional dependencies, the grounding
resource file can also be regenerated locally by running `python -m gilda.generate_terms`.

## Usage
Gilda can either be used programmatically via its Python API, or as a REST service (recommended).
To run the service locally, run
```bash
python -m gilda.app.app
```

This runs the service on port 8001 by default, accepting POST requests on the `http://localhost:8001/ground`
endpoint. The requests use a JSON header, below is an example request using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "kras"}' http://localhost:8001/ground
```

The same request using Python's request package would be as follows:

```python
requests.post('http://localhost:8001/ground', json={'text': 'kras'})
```