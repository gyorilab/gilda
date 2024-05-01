"""
Runs the Gilda grounding app as a module. Usage:

    `python -m gilda.app --host <host> --port <port> --terms <terms>`
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
    args = parse_args()
    app = get_app(args.terms)
    app.run(args.host, args.port, threaded=False)
