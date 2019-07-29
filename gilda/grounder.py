import pandas
import logging
import itertools
from adeft import available_shortforms as available_adeft_models
from adeft.disambiguate import load_disambiguator
from collections import defaultdict
from .term import Term
from .process import normalize, replace_dashes, replace_greek_uni, \
    replace_greek_latin, depluralize
from .scorer import generate_match, score


logger = logging.getLogger(__name__)


class Grounder(object):
    """Class to look up and ground query texts in a terms file."""
    def __init__(self, terms_file):
        self.entries = load_terms_file(terms_file)
        self.disambiguators = load_adeft_models()

    def lookup(self, raw_str):
        """Return matching Terms for a given raw string.

        Parameters
        ----------
        raw_str : str
            A string to be looked up in the set of Terms that the Grounder
            contains.

        Returns
        -------
        list of Term
            A list of Terms that are potential matches for the given string.
        """
        lookups = self._generate_lookups(raw_str)
        entries = []
        for lookup in lookups:
            entries += self.entries.get(lookup, [])
        return entries

    def _generate_lookups(self, raw_str):
        # TODO: we should propagate flags about depluralization and possible
        #  other modifications made here and take them into account when
        #  scoring
        # We first add the normalized string itself
        norm = normalize(raw_str)
        lookups = {norm}
        # Then we add a version with dashes replaced by spaces
        norm_spacedash = normalize(replace_dashes(raw_str, ' '))
        lookups.add(norm_spacedash)
        # We then try to replace spelled out greek letters with
        # their unicode equivalents or their latin equivalents
        greek_replaced = normalize(replace_greek_uni(raw_str))
        lookups.add(greek_replaced)
        greek_replaced = normalize(replace_greek_latin(raw_str))
        lookups.add(greek_replaced)
        # Finally, we attempt to depluralize the word
        depluralized = normalize(depluralize(raw_str)[0])
        lookups.add(depluralized)
        logger.info('Looking up the following strings: %s' %
                    ', '.join(lookups))
        return lookups

    def ground(self, raw_str):
        """Return scored groundings for a given raw string.

        Parameters
        ----------
        raw_str : str
            A string to be grounded with respect to the set of Terms that the
            Grounder contains.

        Returns
        -------
        list of tuple
            A list of tuples with each tuple containing a Term, the score
            associated with the match to that term, and a dict describing
            the match. If a Term was matched multiple times, only the highest
            scoring match is returned.
        """
        entries = self.lookup(raw_str)
        logger.info('Comparing %s with %d entries' %
                    (raw_str, len(entries)))
        # For each entry to compare to, we generate a match data structure
        # describing the comparison of the raw (unnormalized) input string
        # and the entity text corresponding to the matched Term. This match
        # is then further scored to account for the nature of the grounding
        # itself.
        scored_matches = []
        for term in entries:
            match = generate_match(raw_str, term.text)
            sc = score(match, term)
            scored_match = ScoredMatch(term, sc, match)
            scored_matches.append(scored_match)
        unique_scores = self._merge_equivalent_matches(scored_matches)
        return unique_scores

    def disambiguate(self, raw_str, scored_matches, context):
        logger.info('Running disambiguation for %s' % raw_str)
        # If we don't have a disambiguator for this string, we return with
        # the original scores intact
        if raw_str not in self.disambiguators:
            return scored_matches

        # Otherwise, we attempt to disambiguate
        try:
            # We find the disambiguator for the given string and pass in
            # context
            res = self.disambiguators[raw_str].disambiguate([context])
            # The actual grounding dict is at this index in the result
            grounding_dict = res[0][2]
            logger.info('Result from Adeft: %s' % str(grounding_dict))
            # We attempt to get the score for the 'ungrounded' entry
            ungrounded_score = grounding_dict.get('ungrounded', 1.0)
            # Now we check if each scored match has a corresponding Adeft
            # grounding and score. If we find one, we multiply the original
            # match score with the Adeft score. Otherwise, we multiply the
            # original score with the 'ungrounded' score given by Adeft.
            for match in scored_matches:
                has_adeft_grounding = False
                for grounding, score in grounding_dict.items():
                    if grounding == 'ungrounded':
                        continue
                    db, id = grounding.split(':', maxsplit=1)
                    if match.term.db == db and match.term.id == id:
                        match.multiply(score)
                        has_adeft_grounding = True
                        break
                if not has_adeft_grounding:
                    match.multiply(ungrounded_score)
        except Exception as e:
            logger.exception(e)

        return scored_matches

    @staticmethod
    def _merge_equivalent_matches(scored_matches):
        unique_entries = []
        # Characterize an entry by its grounding
        term_dbid = lambda x: (x.term.db, x.term.id)
        # Sort and group scores by grounding
        scored_matches.sort(key=term_dbid)
        entry_groups = itertools.groupby(scored_matches, key=term_dbid)
        # Now look at each group and find the highest scoring match
        for _, entry_group in entry_groups:
            entries = sorted(list(entry_group), key=lambda x: x.score,
                             reverse=True)
            unique_entries.append(entries[0])
        # Return the list of unique entries
        return unique_entries


class ScoredMatch(object):
    """Class representing a scored match to a grounding term.

    Attributes
    -----------
    term : gilda.grounder.Term
        The Term that the scored match is for.
    score : float
        The score associated with the match.
    match : gilda.scorer.Match
        The Match object characterizing the match to the Term.
    """
    def __init__(self, term, score, match):
        self.term = term
        self.score = score
        self.match = match

    def __str__(self):
        return 'ScoredMatch(%s,%s,%s)' % (self.term, self.score, self.match)

    def __repr__(self):
        return str(self)

    def to_json(self):
        return {
            'term': self.term.to_json(),
            'score': self.score,
            'match': self.match.to_json()
        }

    def multiply(self, value):
        logger.info('Multiplying the score of "%s" with %.3f'
                    % (self.term.entry_name, value))
        self.score = self.score * value


def load_terms_file(terms_file):
    """Load a TSV file containing terms into a lookup dictionary.

    Parameters
    ----------
    terms_file : str
        Path to a TSV terms file with columns corresponding to the serialized
        elements of a Term.

    Returns
    -------
    dict
        A lookup dictionary whose keys are normalized entity texts, and values
        are lists of Terms with that normalized entity text.
    """
    df = pandas.read_csv(terms_file, delimiter='\t', na_values=[''],
                         keep_default_na=False)
    entries = defaultdict(list)
    for idx, row in df.iterrows():
        # Replace pandas nans with Nones
        row_nones = [r if not pandas.isna(r) else None for r in row]
        entry = Term(*row_nones)
        entries[row[0]].append(entry)
    return dict(entries)


def load_adeft_models():
    adeft_disambiguators = {}
    for shortform in available_adeft_models:
        adeft_disambiguators[shortform] = load_disambiguator(shortform)
    return adeft_disambiguators
