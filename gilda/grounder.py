import pandas
import logging
import itertools
from .term import Term
from .process import normalize, replace_dashes
from .scorer import generate_match, score


logger = logging.getLogger(__name__)


class Grounder(object):
    def __init__(self, terms_file):
        self.entries = load_terms_file(terms_file)

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
        norm = normalize(raw_str)
        norm_spacedash = normalize(replace_dashes(raw_str, ' '))
        lookups = [norm]
        if norm_spacedash != norm:
            lookups.append(norm_spacedash)
        entries = []
        for lookup in lookups:
            entries += self.entries.get(lookup, [])
        return entries

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
        scores = []
        for entry in entries:
            match = generate_match(raw_str, entry.text)
            score = score(match, entry)
            scores.append((entry, score, match))
        unique_scores = self._merge_equivalent_matches(scores)
        return unique_scores

    @staticmethod
    def _merge_equivalent_matches(scores):
        unique_entries = []
        # Characterize an entry by its grounding
        entry_dbid = lambda x: (x[0].db, x[0].id)
        # Sort and group scores by grounding
        scores.sort(key=entry_dbid)
        entry_groups = itertools.groupby(scores, key=entry_dbid)
        # Now look at each group and find the highest scoring match
        for _, entry_group in entry_groups:
            entries = sorted(list(entry_group), key=lambda x: x[1],
                             reverse=True)
            unique_entries.append(entries[0])
        # Return the list of unique entries
        return unique_entries


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
    entries = {}
    for idx, row in df.iterrows():
        # Replace pandas nans with Nones
        row_nones = [r if not pandas.isna(r) else None for r in row]
        entry = Term(*row_nones)
        if row[0] in entries:
            entries[row[0]].append(entry)
        else:
            entries[row[0]] = [entry]
    return entries
