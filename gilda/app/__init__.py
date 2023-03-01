import argparse
from .app import get_app


def main():
    parser = argparse.ArgumentParser(
        description='Run the grounding app.')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=8001, type=int)
    parser.add_argument('--terms')
    args = parser.parse_args()
    app = get_app(terms=args.terms)
    app.run(host=args.host, port=args.port, threaded=False)
