"""
Gilda implements a simple dictionary-based named entity
recognition (NER) algorithm. It can be used as follows:

>>> from gilda.ner import annotate
>>> text = "MEK phosphorylates ERK"
>>> results = annotate(text)

The results are a list of 4-tuples containing:

- the text string matched
- a :class:`gilda.grounder.ScoredMatch` instance containing the _best_ match
- the position in the text string where the entity starts
- the position in the text string where the entity ends

In this example, the two concepts are grounded to FamPlex entries.

>>> results[0][0], results[0][1].term.get_curie(), results[0][2], results[0][3]
('MEK', 'fplx:MEK', 0, 3)
>>> results[1][0], results[1][1].term.get_curie(), results[1][2], results[1][3]
('ERK', 'fplx:ERK', 19, 22)

If you directly look in the second part of the 4-tuple, you get a full
description of the match itself:

>>> results[0][1]
ScoredMatch(Term(mek,MEK,FPLX,MEK,MEK,curated,famplex,None,None,None),\
0.9288806431663574,Match(query=mek,ref=MEK,exact=False,space_mismatch=\
False,dash_mismatches=set(),cap_combos=[('all_lower', 'all_caps')]))

BRAT
----
Gilda implements a way to output annotation in a format appropriate for the
`BRAT Rapid Annotation Tool (BRAT) <https://brat.nlplab.org/index.html>`_.

>>> from gilda.ner import get_brat
>>> from pathlib import Path
>>> brat_string = get_brat(results)
>>> Path("results.ann").write_text(brat_string)
>>> Path("results.txt").write_text(text)

For brat to work, you need to store the text in a file with
the extension ``.txt`` and the annotations in a file with the
same name but extension ``.ann``.
"""

from typing import List, Tuple

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from gilda import ScoredMatch, get_grounder
from gilda.process import normalize

__all__ = [
    "annotate",
    "get_brat",
    "Annotation",
]

stop_words = set(stopwords.words('english'))

Annotation = Tuple[str, ScoredMatch, int, int]


def annotate(
    text, *,
    grounder=None,
    sent_split_fun=None,
    organisms=None,
    namespaces=None,
    return_first: bool = True,
) -> List[Annotation]:
    """Annotate a given text with Gilda.

    Parameters
    ----------
    text : str
        The text to be annotated.
    grounder : gilda.grounder.Grounder, optional
        The Gilda grounder to use for grounding.
    sent_split_fun : Callable, optional
        A function that splits the text into sentences. The default is
        :func:`nltk.tokenize.sent_tokenize`. The function should take a string
        as input and return an iterable of strings corresponding to the sentences
        in the input text.
    organisms : list[str], optional
        A list of organism names to pass to the grounder. If not provided,
        human is used.
    namespaces : list[str], optional
        A list of namespaces to pass to the grounder to restrict the matches
        to. By default, no restriction is applied.
    return_first:
        If true, only returns the first result. Otherwise, returns all results.

    Returns
    -------
    list[tuple[str, ScoredMatch, int, int]]
        A list of tuples of start and end character offsets of the text
        corresponding to the entity, the entity text, and the ScoredMatch
        object corresponding to the entity.
    """
    if grounder is None:
        grounder = get_grounder()
    if sent_split_fun is None:
        sent_split_fun = sent_tokenize
    # Get sentences
    sentences = sent_split_fun(text)
    text_coord = 0
    entities = []
    for sentence in sentences:
        raw_words = [w for w in sentence.rstrip('.').split()]
        word_coords = [text_coord]
        for word in raw_words:
            word_coords.append(word_coords[-1] + len(word) + 1)
        text_coord += len(sentence) + 1
        words = [normalize(w) for w in raw_words]
        skip_until = 0
        for idx, word in enumerate(words):
            if idx < skip_until:
                continue
            if word in stop_words:
                continue
            spans = grounder.prefix_index.get(word, set())
            if not spans:
                continue

            # Only consider spans that are within the sentence
            applicable_spans = {span for span in spans
                                if idx + span <= len(words)}

            # Find the largest matching span
            for span in sorted(applicable_spans, reverse=True):
                txt_span = ' '.join(words[idx:idx+span])
                matches = grounder.ground(
                    txt_span, context=text,
                    organisms=organisms, namespaces=namespaces,
                )
                if matches:
                    start_coord = word_coords[idx]
                    end_coord = word_coords[idx+span-1] + \
                        len(raw_words[idx+span-1])
                    raw_span = ' '.join(raw_words[idx:idx+span])

                    if return_first:
                        matches = [matches[0]]
                    for match in matches:
                        entities.append(
                            (raw_span, match, start_coord, end_coord)
                        )

                    skip_until = idx + span
                    break
    return entities


def get_brat(entities, entity_type="Entity", ix_offset=1, include_text=True):
    """Return brat-formatted annotation strings for the given entities.

    Parameters
    ----------
    entities : list[tuple[str, str | ScoredMatch, int, int]]
        A list of tuples of entity text, grounded curie, start and end
        character offsets in the text corresponding to an entity.
    entity_type : str, optional
        The brat entity type to use for the annotations. The default is
        'Entity'. This is useful for differentiating between annotations in
        the same text extracted from different reading systems.
    ix_offset : int, optional
        The index offset to use for the brat annotations. The default is 1.
    include_text : bool, optional
        Whether to include the text of the entity in the brat annotations.
        The default is True. If not provided, the text that matches the span
        will be written to the annotation file.

    Returns
    -------
    str
        A string containing the brat-formatted annotations.
    """
    brat = []
    ix_offset = max(1, ix_offset)
    for idx, (raw_span, curie, start, end) in enumerate(entities, ix_offset):
        if isinstance(curie, ScoredMatch):
            curie = curie.term.get_curie()
        if entity_type != "Entity":
            curie += f"; Reading system: {entity_type}"
        row = f'T{idx}\t{entity_type} {start} {end}' + (
            f'\t{raw_span}' if include_text else ''
        )
        brat.append(row)
        row = f'#{idx}\tAnnotatorNotes T{idx}\t{curie}'
        brat.append(row)
    return '\n'.join(brat) + '\n'
