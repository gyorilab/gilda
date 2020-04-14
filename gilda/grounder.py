import csv
import json
import pickle
import logging
import itertools
from adeft.disambiguate import load_disambiguator
from adeft.modeling.classify import load_model_info
from adeft import available_shortforms as available_adeft_models
from .term import Term
from .process import normalize, replace_dashes, replace_greek_uni, \
    replace_greek_latin, depluralize
from .scorer import generate_match, score, score_namespace
from .resources import get_gilda_models, get_grounding_terms


logger = logging.getLogger(__name__)


class Grounder(object):
    """Class to look up and ground query texts in a terms file.

    Parameters
    ----------
    terms : str or dict or None
        Specifies the grounding terms that should be loaded in the Grounder.
        If None, the default grounding terms are loaded from the versioned
        resource folder. If str, it is interpreted as a path to a grounding
        terms TSV file which is then loaded. If dict, it is assumed to be
        a grounding terms dict with normalized entity strings as keys
        and Term objects as values. Default: None
    """
    def __init__(self, terms=None):
        if terms is None:
            terms = get_grounding_terms()

        if isinstance(terms, str):
            self.entries = load_terms_file(terms)
        elif isinstance(terms, dict):
            self.entries = terms
        else:
            raise TypeError('terms is neither a path nor a normalized'
                            ' entry name to term dictionary')

        self.adeft_disambiguators = load_adeft_models()
        self.gilda_disambiguators = load_gilda_models()

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
        logger.debug('Looking up the following strings: %s' %
                     ', '.join(lookups))
        return lookups

    def ground(self, raw_str, context=None):
        """Return scored groundings for a given raw string.

        Parameters
        ----------
        raw_str : str
            A string to be grounded with respect to the set of Terms that the
            Grounder contains.
        context : Optional[str]
            Any additional text that serves as context for disambiguating the
            given entity text, used if a model exists for disambiguating the
            given text.

        Returns
        -------
        list[gilda.grounder.ScoredMatch]
            A list of ScoredMatch objects representing the groundings sorted
            by decreasing score.
        """
        entries = self.lookup(raw_str)
        logger.debug('Comparing %s with %d entries' %
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

        # Return early if we don't have anything to avoid calling other
        # functions with no matches
        if not scored_matches:
            return scored_matches

        # Merge equivalent matches
        unique_scores = self._merge_equivalent_matches(scored_matches)

        # If there's context available, disambiguate based on that
        if context:
            unique_scores = self.disambiguate(raw_str, unique_scores, context)

        # Then sort by decreasing score
        rank_fun = lambda x: (x.score, score_namespace(x.term))
        unique_scores = sorted(unique_scores, key=rank_fun, reverse=True)

        return unique_scores

    def disambiguate(self, raw_str, scored_matches, context):
        # If we don't have a disambiguator for this string, we return with
        # the original scores intact. Otherwise, we attempt to disambiguate.
        if raw_str in self.adeft_disambiguators:
            logger.info('Running Adeft disambiguation for %s' % raw_str)
            try:
                scored_matches = \
                    self.disambiguate_adeft(raw_str, scored_matches, context)
            except Exception as e:
                logger.exception(e)
        elif raw_str in self.gilda_disambiguators:
            logger.info('Running Gilda disambiguation for %s' % raw_str)
            try:
                scored_matches = \
                    self.disambiguate_gilda(raw_str, scored_matches, context)
            except Exception as e:
                logger.exception(e)

        return scored_matches

    def disambiguate_adeft(self, raw_str, scored_matches, context):
        # We find the disambiguator for the given string and pass in
        # context
        res = self.adeft_disambiguators[raw_str].disambiguate([context])
        # The actual grounding dict is at this index in the result
        grounding_dict = res[0][2]
        logger.debug('Result from Adeft: %s' % str(grounding_dict))
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
                    match.disambiguation = {'type': 'adeft',
                                            'score': score,
                                            'match': 'grounded'}
                    match.multiply(score)
                    has_adeft_grounding = True
                    break
            if not has_adeft_grounding:
                match.disambiguation = {'type': 'adeft',
                                        'score': ungrounded_score,
                                        'match': 'ungrounded'}
                match.multiply(ungrounded_score)
        return scored_matches

    def disambiguate_gilda(self, raw_str, scored_matches, context):
        res = self.gilda_disambiguators[raw_str].predict_proba([context])
        if not res:
            raise ValueError('No result from disambiguation.')
        grounding_dict = res[0]
        for match in scored_matches:
            key = '%s:%s' % (match.term.db, match.term.id)
            score_entry = grounding_dict.get(key, None)
            score = score_entry if score_entry is not None else 0.0
            match.disambiguation = {'type': 'gilda',
                                    'score': score,
                                    'match': ('grounded'
                                              if score_entry is not None
                                              else 'ungrounded')}
            match.multiply(score)
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

    def get_models(self):
        """Return a list of entity texts for which disambiguation models exist.

        Returns
        -------
        list[str]
            The list of entity texts for which a disambiguation model is
            available.
        """
        return sorted(list(self.gilda_disambiguators.keys()))

    def get_names(self, db, id, status=None, source=None):
        """Return a list of entity texts corresponding to a given database ID.

        Parameters
        ----------
        db : str
            The database in which the ID is an entry, e.g., HGNC.
        id : str
            The ID of an entry in the database.
        status : Optional[str]
            If given, only entity texts with the given status e.g., "synonym"
            are returned.
        source : Optional[str]
            If given, only entity texts from the given source e.g., "uniprot"
            are returned.

        Returns
        -------
        names: list[str]
            A list of entity texts corresponding to the given database/ID
        """
        names = set()
        for entries in self.entries.values():
            for entry in entries:
                if (entry.db == db) and (entry.id == id) and \
                   (not status or entry.status == status) and \
                   (not source or entry.source == source):
                    names.add(entry.text)
        return sorted(names)


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
    disambiguation : Optional[dict]
        Meta-information about disambiguation, when available.
    """
    def __init__(self, term, score, match, disambiguation=None):
        self.term = term
        self.score = score
        self.match = match
        self.disambiguation = disambiguation

    def __str__(self):
        disamb_str = '' if self.disambiguation is None else \
            (',disambiguation=' + json.dumps(self.disambiguation))
        return 'ScoredMatch(%s,%s,%s%s)' % \
            (self.term, self.score, self.match, disamb_str)

    def __repr__(self):
        return str(self)

    def to_json(self):
        js = {
            'term': self.term.to_json(),
            'score': self.score,
            'match': self.match.to_json()
        }
        if self.disambiguation is not None:
            js['disambiguation'] = self.disambiguation
        return js

    def multiply(self, value):
        logger.debug('Multiplying the score of "%s" with %.3f'
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
    with open(terms_file, 'r') as fh:
        entries = {}
        for row in csv.reader(fh, delimiter='\t'):
            row_nones = [r if r else None for r in row]
            entry = Term(*row_nones)
            if row[0] in entries:
                entries[row[0]].append(entry)
            else:
                entries[row[0]] = [entry]
        return entries


def load_adeft_models():
    adeft_disambiguators = {}
    for shortform in available_adeft_models:
        adeft_disambiguators[shortform] = load_disambiguator(shortform)
    return adeft_disambiguators


def load_gilda_models():
    with open(get_gilda_models(), 'rb') as fh:
        models_raw = pickle.load(fh)
    models = {k: load_model_info(v['cl']) for k, v in models_raw.items()}
    models = {k: v for k, v in models.items() if v.stats['f1']['mean'] > 0.7}
    return models
