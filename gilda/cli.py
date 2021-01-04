# -*- coding: utf-8 -*-

"""Command line interface for Gilda.

Why does this file exist, and why not put this in ``__main__``? You might be tempted to import things from ``__main__``
later, but that will cause problems--the code will get executed twice:

- When you run ``python3 -m gilda`` python will execute``__main__.py`` as a script. That means there won't be any
  ``gilda.__main__`` in ``sys.modules``.
- When you import __main__ it will get executed again (as a module) because
  there's no ``gilda.__main__`` in ``sys.modules``.

.. seealso:: https://click.palletsprojects.com/en/7.x/setuptools/#setuptools-integration
"""

import json
from typing import Optional

import click


@click.group()
def main():
    """Gilda CLI."""


@main.command()
@click.argument('text')
@click.option('-c', '--context')
def ground(text: str, context: Optional[str]):
    """Ground the term"""
    from gilda.api import ground
    for scored_match in ground(text=text, context=context):
        click.echo(json.dumps(scored_match.to_json()))


@main.command()
def web():
    """Run the Gilda web app."""
    from .app.app import app
    app.run()


if __name__ == '__main__':
    main()
