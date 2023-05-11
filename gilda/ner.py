from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from gilda.process import normalize

stop_words = set(stopwords.words('english'))


def annotate(grounder, text):
    sentences = sent_tokenize(text)
    text_coord = 0
    entities = []
    for sentence in sentences:
        raw_words = [w for w in sentence.rstrip('.').split()]
        word_coords = [text_coord]
        for word in raw_words:
            word_coords.append(word_coords[-1] + len(word) + 1)
        text_coord += len(sentence) + 1
        words = [normalize(w) for w in raw_words]
        skip_until = None
        for idx, word in enumerate(words):
            if skip_until is not None and idx < skip_until:
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
