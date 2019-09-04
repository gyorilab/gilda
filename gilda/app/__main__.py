import argparse
from . import app


def main():
    parser = argparse.ArgumentParser(
        description='Run the grounding app.')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=8001, type=int)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == '__main__':
    main()
