import csv
import json

import click
import pyobo
import pystow
from more_click import verbose_option
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import gilda
from pubtator_loader import from_gz

URL = "https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator.txt.gz"
MODULE = pystow.module("gilda", "medmentions")
CORPUS_PATH = MODULE.join(name="corpus.json")
MATCHING_PATH = MODULE.join(name="matching.tsv")


def get_corpus():
    """Get the MedMentions corpus.

    :return: A list of dictionaries of the following form:

    .. code-block::

        [
          {
            "id": 25763772,
            "title_text": "DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis",
            "abstract_text": "...",
            "entities": [
              {
                "document_id": 25763772,
                "start_index": 0,
                "end_index": 5,
                "text_segment": "DCTN4",
                "semantic_type_id": "T116,T123",
                "entity_id": "C4308010"
              },
              ...
            ]
          },
          ...
        ]
    """
    if not CORPUS_PATH.is_file():
        path = MODULE.ensure(url=URL)
        corpus = from_gz(path)
        with CORPUS_PATH.open("w") as file:
            json.dump(corpus, file, indent=2, default=lambda o: o.__dict__)

    # Right now I'd rather not engage with the strange object structure, so
    # serializing and deserializing gets us JSON we can work with.
    with CORPUS_PATH.open() as file:
        return json.load(file)


@click.command()
@verbose_option
def main():
    corpus = get_corpus()
    click.echo(f"There are {len(corpus)} entries")
    rows = []
    for document in tqdm(corpus, desc="Documents"):
        abstract = document["abstract_text"]
        for entity in document["entities"]:
            umls_id = entity["entity_id"]
            text = entity["text_segment"]
            with logging_redirect_tqdm():
                matches = gilda.ground(text, context=abstract)
            for match in matches:
                rows.append(
                    (
                        umls_id,
                        pyobo.get_name("umls", umls_id),
                        text,
                        match.term.db,
                        match.term.id,
                        match.term.entry_name,
                        match.score,
                    )
                )
    with MATCHING_PATH.open("w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(
            (
                "umls_id",
                "umls_name",
                "text",
                "gilda_prefix",
                "gilda_identifier",
                "gilda_name",
                "gilda_score",
            )
        )
        writer.writerows(rows)


if __name__ == "__main__":
    main()
