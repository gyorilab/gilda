"""
Run the grounding app.

Run as module:
    `python -m gilda.app --host <host> --port <port> --terms <terms>`

Run with gunicorn:
    `gunicorn -w <worker count> -b <host>:<port> -t <timeout> gilda.app:gilda_app`

In case a non-standard set of terms is to be used, set the `GILDA_TERMS`
environment variable to the path to the terms file.
"""

import os
from .app import get_app


terms = os.environ.get('GILDA_TERMS')
gilda_app = get_app(terms=terms)
