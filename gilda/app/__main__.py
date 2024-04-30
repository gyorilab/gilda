"""
Runs the Gilda grounding app as a module. Usage:

    `python -m gilda.app --host <host> --port <port> --terms <terms>`
"""

from .app import get_app
from . import parse_args


if __name__ == '__main__':
    args = parse_args()
    app = get_app(args.terms)
    app.run(args.host, args.port, threaded=False)
