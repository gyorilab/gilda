from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

from gilda import ScoredMatch
from gilda.process import normalize

stop_words = set(stopwords.words('english'))


def annotate(grounder, text, sent_split_fun=sent_tokenize):
    """Annotate a given text with Gilda.

    Parameters
    ----------
    grounder : gilda.grounder.Grounder
        The Gilda grounder to use for grounding.
    text : str
        The text to be annotated.
    sent_split_fun : Callable, optional
        A function that splits the text into sentences. The default is
        nltk.tokenize.sent_tokenize. The function should take a string as
        input and return an iterable of strings corresponding to the sentences
        in the input text.

    Returns
    -------
    list[tuple[str, ScoredMatch, int, int]]
        A list of tuples of start and end character offsets of the text
        corresponding to the entity, the entity text, and the ScoredMatch
        object corresponding to the entity.
    """
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
                matches = grounder.ground(txt_span)
                if matches:
                    start_coord = word_coords[idx]
                    end_coord = word_coords[idx+span-1] + \
                        len(raw_words[idx+span-1])
                    raw_span = ' '.join(raw_words[idx:idx+span])

                    # Append raw_span, (best) match, start, end
                    match = matches[0]
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
