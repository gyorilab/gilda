import json

import click
import pystow
from more_click import verbose_option

from pubtator_loader import from_gz

URL = 'https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator.txt.gz'
MODULE = pystow.module('gilda', 'medmentions')
CORPUS_PATH = MODULE.join(name='corpus.json')


def get_corpus():
    if not CORPUS_PATH.is_file():
        path = MODULE.ensure(url=URL)
        corpus = from_gz(path)
        with CORPUS_PATH.open('w') as file:
            json.dump(corpus, file, indent=2, default=lambda o: o.__dict__)

    # Right now I'd rather not engage with the strange object structure, so
    # serializing and deserializing gets us JSON we can work with.
    with CORPUS_PATH.open() as file:
        return json.load(file)


@click.command()
@verbose_option
def main():
    corpus = get_corpus()
    click.echo(f'There are {len(corpus)} entries')


if __name__ == '__main__':
    main()
