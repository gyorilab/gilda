"""
Gilda implements a simple dictionary-based named entity
recognition (NER) algorithm. It can be used as follows:

>>> from gilda.ner import annotate
>>> text = "MEK phosphorylates ERK"
>>> results = annotate(text)

The results are a list of Annotation objects each of which contains:

- the `text` string matched
- a list of :class:`gilda.grounder.ScoredMatch` instances containing a sorted list of matches
  for the given text span (first one is the best match)
- the `start` position in the text string where the entity starts
- the `end` position in the text string where the entity ends


In this example, the two concepts are grounded to FamPlex entries.

>>> results[0].text, results[0].matches[0].term.get_curie(), results[0].start, results[0].end
('MEK', 'fplx:MEK', 0, 3)
>>> results[1].text, results[1].matches[0].term.get_curie(), results[1].start, results[1].end
('ERK', 'fplx:ERK', 19, 22)

If you directly look in the second part of the 4-tuple, you get a full
description of the match itself:

>>> results[0].matches[0]
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

from typing import List, Set
import os

from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer

from gilda import get_grounder
from gilda.grounder import Annotation
from gilda.process import normalize

__all__ = [
    "annotate",
    "get_brat",
    "stop_words"
]

STOPLIST_PATH = os.path.join(os.path.dirname(__file__),'resources',
                             'ner_stoplist.txt')


def _load_stoplist() -> Set[str]:
    """Load NER stoplist from file."""
    stoplist_path = STOPLIST_PATH
    with open(stoplist_path, 'r') as file:
        stoplist = {line.strip() for line in file}
    return stoplist


stop_words = set(stopwords.words('english'))
stop_words.update(_load_stoplist())


def annotate(
    text, *,
    grounder=None,
    sent_split_fun=None,
    organisms=None,
    namespaces=None,
    context_text: str = None,
) -> List[Annotation]:
    """Annotate a given text with Gilda.

    Parameters
    ----------
    text : str
        The text to be annotated.
    grounder : gilda.grounder.Grounder, optional
        The Gilda grounder to use for grounding.
    sent_split_fun : Callable[str, Iterable[Tuple[int, int]]], optional
        A function that splits the text into sentences. The default is
        :func:`nltk.tokenize.PunktSentenceTokenizer.span_tokenize`. The function
        should take a string as input and return an iterable of coordinate pairs
        corresponding to the start and end coordinates for each sentence in the
        input text.
    organisms : list[str], optional
        A list of organism names to pass to the grounder. If not provided,
        human is used.
    namespaces : List[str], optional
        A list of namespaces to pass to the grounder to restrict the matches
        to. By default, no restriction is applied.
    context_text :
        A longer span of text that serves as additional context for the text
        being annotated for disambiguation purposes.

    Returns
    -------
    List[Annotation]
        A list of Annotations where each contains as attributes
        the text span that was matched, the list of ScoredMatches, and the
        start and end character offsets of the text span.
    """
    if grounder is None:
        grounder = get_grounder()
    if sent_split_fun is None:
        sent_tokenizer = PunktSentenceTokenizer()
        sent_split_fun = sent_tokenizer.span_tokenize
    # Get sentences
    sentence_coords = sent_split_fun(text)
    text_coord = 0
    annotations = []
    word_tokenizer = TreebankWordTokenizer()
    # FIXME: a custom sentence split function can be inconsistent
    # with the coordinates being used here which come from NLTK
    for sent_start, sent_end in sentence_coords:
        sentence = text[sent_start:sent_end]
        # FIXME: one rare corner case is named entities with single quotes
        # in them which get tokenized in a weird way
        raw_word_coords = \
            list(word_tokenizer.span_tokenize(sentence.rstrip('.')))
        raw_words = [sentence[start:end] for start, end in raw_word_coords]
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
                # We have to reconstruct a text span while adding spaces
                # where needed
                raw_span = ''
                for rw, c in zip(raw_words[idx:idx+span],
                                    raw_word_coords[idx:idx+span]):
                    # Figure out if we need a space before this word, then
                    # append the word.
                    spaces = ' ' * (c[0] - len(raw_span) -
                                    raw_word_coords[idx][0])
                    raw_span += spaces + rw
                # If span is a single character, we don't want to consider it
                if len(raw_span) <= 1:
                    continue
                context = text if context_text is None else context_text
                matches = grounder.ground(raw_span,
                                          context=context,
                                          organisms=organisms,
                                          namespaces=namespaces)
                if matches:
                    start_coord = sent_start + raw_word_coords[idx][0]
                    end_coord = sent_start + raw_word_coords[idx+span-1][1]
                    annotations.append(Annotation(
                        raw_span, matches, start_coord, end_coord
                    ))

                    skip_until = idx + span
                    break
    return annotations


def get_brat(annotations, entity_type="Entity", ix_offset=1, include_text=True):
    """Return brat-formatted annotation strings for the given entities.

    Parameters
    ----------
    annotations : list[Annotation]
        A list of named entity annotations in the text.
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
    for idx, annotation in enumerate(annotations, ix_offset):
        curie = annotation.matches[0].term.get_curie()
        if entity_type != "Entity":
            curie += f"; Reading system: {entity_type}"
        row = f'T{idx}\t{entity_type} {annotation.start} {annotation.end}' + (
            f'\t{annotation.text}' if include_text else ''
        )
        brat.append(row)
        row = f'#{idx}\tAnnotatorNotes T{idx}\t{curie}'
        brat.append(row)
    return '\n'.join(brat) + '\n'
