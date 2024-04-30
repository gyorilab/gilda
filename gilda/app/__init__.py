"""
Run the grounding app.

Run as module:
    `python -m gilda.app --host <host> --port <port> --terms <terms>`

Run with gunicorn:
    `gunicorn -w <worker count> -b <host>:<port> -t <timeout> gilda.app:gunicorn_app`
"""

import argparse
from .app import get_app


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the grounding app.')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=8001, type=int)
    parser.add_argument('--terms')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    _args = parse_args()
    app = get_app(_args.terms)
    app.run(_args.host, _args.port, threaded=False)
else:
    gunicorn_app = get_app()
