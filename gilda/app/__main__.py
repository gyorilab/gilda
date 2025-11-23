"""
Runs the Gilda grounding app as a module. Usage:

    `python -m gilda.app --host <host> --port <port> --terms <terms>`
"""
from .app import main


if __name__ == "__main__":
    main()
