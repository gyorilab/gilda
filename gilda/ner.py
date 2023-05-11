from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
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
    list[tuple[int, int, str, ScoredMatch]]
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
            applicable_spans = {span for span in spans
                                if idx + span <= len(words)}
            for span in sorted(applicable_spans, reverse=True):
                txt_span = ' '.join(words[idx:idx+span])
                matches = grounder.ground(txt_span)
                if matches:
                    start_coord = word_coords[idx]
                    end_coord = word_coords[idx+span-1] + \
                        len(raw_words[idx+span-1])
                    raw_span = ' '.join(raw_words[idx:idx+span])
                    entities.append((start_coord, end_coord,
                                     raw_span, matches))
                    skip_until = idx + span
                    break
    return entities


def get_brat(entities):
    brat = []
    for idx, (start, end, raw_span, matches) in enumerate(entities, 1):
        match = matches[0]
        grounding = match.term.db + ':' + match.term.id
        row = f'T{idx}\tEntity {start} {end}\t{raw_span}'
        brat.append(row)
        row = f'#{idx}\tAnnotatorNotes T{idx}\t{grounding}'
        brat.append(row)
    return '\n'.join(brat)
