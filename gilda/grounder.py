import os
import csv
import json
import gzip
import logging
import itertools
import collections.abc
from pathlib import Path
from collections import defaultdict, Counter
from textwrap import dedent
from typing import Iterator, List, Mapping, Optional, Set, Tuple, Union, Iterable
from adeft.disambiguate import load_disambiguator
from adeft.modeling.classify import load_model_info
from adeft import available_shortforms as available_adeft_models
from .term import Term, get_identifiers_curie, get_identifiers_url
from .process import normalize, replace_dashes, replace_greek_uni, \
    replace_greek_latin, replace_greek_spelled_out, depluralize, \
    replace_roman_arabic
from .scorer import Match, generate_match, score
from .resources import get_gilda_models, get_grounding_terms

__all__ = [
    "Grounder",
    "GrounderInput",
    "ScoredMatch",
    "load_terms_file",
    "load_entries_from_terms_file",
    "filter_for_organism",
    "load_adeft_models",
    "load_gilda_models",
]

logger = logging.getLogger(__name__)


GrounderInput = Union[str, Path, Iterable[Term], Mapping[str, List[Term]]]

#: The default namespace priority order
DEFAULT_NAMESPACE_PRIORITY = [
    'FPLX', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'DOID', 'HP', 'EFO'
]


class Grounder(object):
    """Class to look up and ground query texts in a terms file.

    Parameters
    ----------
    terms :
        Specifies the grounding terms that should be loaded in the Grounder.

        - If ``None``, the default grounding terms are loaded from the
          versioned resource folder.
        - If :class:`str` or :class:`pathlib.Path`, it is interpreted
          as a path to a grounding terms gzipped TSV file which is then
          loaded.
        - If :class:`dict`, it is assumed to be a grounding terms dict with
          normalized entity strings as keys and :class:`gilda.term.Term`
          instances as values.
        - If :class:`list`, :class:`set`, :class:`tuple`, or any other iterable,
          it is assumed to be a flat list of
          :class:`gilda.term.Term` instances.
    namespace_priority :
        Specifies a term namespace priority order. For example, if multiple
        terms are matched with the same score, will use this list to decide
        which are given by which namespace appears further towards the front
        of the list. By default, :data:`DEFAULT_NAMESPACE_PRIORITY` is used,
        which, for example, prioritizes famplex entities over HGNC ones.
    """

    entries: Mapping[str, List[Term]]
    namespace_priority: List[str]

    def __init__(
        self,
        terms: Optional[GrounderInput] = None,
        *,
        namespace_priority: Optional[List[str]] = None,
    ):
        if terms is None:
            terms = get_grounding_terms()

        if isinstance(terms, (str, Path)):
            extension = os.path.splitext(terms)[1]
            if extension == '.db':
                from .resources.sqlite_adapter import SqliteEntries
                self.entries = SqliteEntries(terms)
            else:
                self.entries = load_terms_file(terms)
        elif isinstance(terms, dict):
            self.entries = terms
        elif isinstance(terms, collections.abc.Iterable):
            self.entries = defaultdict(list)
            for term in terms:
                self.entries[term.norm_text].append(term)
            self.entries = dict(self.entries)
        else:
            raise TypeError('terms is neither a path nor a list of terms,'
                            'nor a normalized entry name to term dictionary')

        self.prefix_index = {}
        self._build_prefix_index()

        self.adeft_disambiguators = find_adeft_models()
        self.gilda_disambiguators = None

        self.namespace_priority = (
            DEFAULT_NAMESPACE_PRIORITY
            if namespace_priority is None else
            namespace_priority
        )

    def _build_prefix_index(self):
        prefix_index = defaultdict(set)
        for norm_term in self.entries:
            if not norm_term:
                continue
            parts = norm_term.split()
            if not parts:
                continue
            prefix_index[parts[0]].add(len(parts))
        self.prefix_index = dict(prefix_index)

    def lookup(self, raw_str: str) -> List[Term]:
        """Return matching Terms for a given raw string.

        Parameters
        ----------
        raw_str :
            A string to be looked up in the set of Terms that the Grounder
            contains.

        Returns
        -------
        :
            A list of Terms that are potential matches for the given string.
        """
        lookups = self._generate_lookups(raw_str)
        entries = []
        for lookup in lookups:
            entries += self.entries.get(lookup, [])
        return entries

    def _generate_lookups(self, raw_str: str) -> Set[str]:
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
        greek_replaced = normalize(replace_greek_spelled_out(raw_str))
        lookups.add(greek_replaced)
        # We try exchanging roman and arabic numerals
        roman_arabic = normalize(replace_roman_arabic(raw_str))
        lookups.add(roman_arabic)
        # Finally, we attempt to depluralize the word
        for singular, rule in depluralize(raw_str):
            lookups.add(normalize(singular))

        logger.debug('Looking up the following strings: %s' %
                     ', '.join(lookups))
        return lookups

    def _score_namespace(self, term) -> int:
        """Apply a priority to the term based on its namespace.

        .. note::

            This is currently not included as an explicit score term.
            It is just used to rank identically scored entries.
        """
        try:
            return len(self.namespace_priority) - self.namespace_priority.index(term.db)
        except ValueError:
            return 0

    def ground_best(
        self,
        raw_str: str,
        context: Optional[str] = None,
        organisms: Optional[List[str]] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Optional["ScoredMatch"]:
        """Return the best scored grounding for a given raw string.

        Parameters
        ----------
        raw_str : str
            A string to be grounded with respect to the set of Terms that the
            Grounder contains.
        context : Optional[str]
            Any additional text that serves as context for disambiguating the
            given entity text, used if a model exists for disambiguating the
            given text.
        organisms : Optional[List[str]]
            An optional list of organism identifiers defining a priority
            ranking among organisms, if genes/proteins from multiple
            organisms match the input. If not provided, the default
            ['9606'] i.e., human is used.
        namespaces : Optional[List[str]]
            A list of namespaces to restrict matches to. This will apply to
            both the primary namespace of a matched term, to any subsumed
            matches, and to the source namespaces of terms if they were
            created using cross-reference mappings. By default, no
            restriction is applied.

        Returns
        -------
        Optional[gilda.grounder.ScoredMatch]
            The best ScoredMatch returned by :meth:`ground` if any are returned,
            otherwise None.
        """
        scored_matches = self.ground(
            raw_str=raw_str,
            context=context,
            organisms=organisms,
            namespaces=namespaces,
        )
        if scored_matches:
            # Because of the way the ground() function is implemented,
            # the first element is guaranteed to have the best score
            # (after filtering by namespace)
            return scored_matches[0]
        return None

    def ground(self, raw_str, context=None, organisms=None,
               namespaces=None):
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
        organisms : Optional[List[str]]
            An optional list of organism identifiers defining a priority
            ranking among organisms, if genes/proteins from multiple
            organisms match the input. If not provided, the default
            ['9606'] i.e., human is used.
        namespaces : Optional[List[str]]
            A list of namespaces to restrict matches to. This will apply to
            both the primary namespace of a matched term, to any subsumed
            matches, and to the source namespaces of terms if they were
            created using cross-reference mappings. By default, no
            restriction is applied.

        Returns
        -------
        list[gilda.grounder.ScoredMatch]
            A list of ScoredMatch objects representing the groundings sorted
            by decreasing score.
        """
        if not organisms:
            organisms = ['9606']
        # Stripping whitespaces is done up front directly on the raw string
        # so that all lookups and comparisons are done with respect to the
        # stripped string
        raw_str = raw_str.strip()
        # Initial lookup of all possible matches
        entries = self.lookup(raw_str)
        logger.debug('Filtering %d entries by organism' % len(entries))
        entries = filter_for_organism(entries, organisms)
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
        rank_fun = lambda x: (x.score, self._score_namespace(x.term))
        unique_scores = sorted(unique_scores, key=rank_fun, reverse=True)

        # If we have a namespace constraint, we filter to the given
        # namespaces.
        if namespaces:
            unique_scores = [
                scored_match for scored_match in unique_scores
                if scored_match.get_namespaces() & set(namespaces)
            ]

        return unique_scores

    def disambiguate(self, raw_str, scored_matches, context):
        # This is only called if context was passed in so we do lazy
        # loading here
        if self.gilda_disambiguators is None:
            self.gilda_disambiguators = load_gilda_models()
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
        if self.adeft_disambiguators[raw_str] is None:
            self.adeft_disambiguators[raw_str] = load_disambiguator(raw_str)
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
                # There is a corner case here where grounding is
                # some name other than 'ungrounded' but is not a proper
                # ns:id pair.
                if grounding == 'ungrounded' or ':' not in grounding:
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
            entries[0].subsumed_terms = [e.term for e in entries[1:]]
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
        if self.gilda_disambiguators is None:
            self.gilda_disambiguators = load_gilda_models()
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

    def get_ambiguities(self,
                        skip_names: bool = True,
                        skip_curated: bool = True,
                        skip_name_matches: bool = True,
                        skip_species_ambigs: bool = True) -> List[List[Term]]:
        """Return a list of ambiguous term groups in the grounder.

        Parameters
        ----------
        skip_names :
            If True, groups of terms where one has the "name" status are
            skipped. This makes sense usually since these are prioritized over
            synonyms anyway.
        skip_curated :
            If True, groups of terms where one has the "curated" status
            are skipped. This makes sense usually since these are prioritized
            over synonyms anyway.
        skip_name_matches :
            If True, groups of terms that all share the same standard name
            are skipped. This is effective at eliminating spurious ambiguities
            due to unresolved cross-references between equivalent terms
            in different namespaces.
        skip_species_ambigs :
            If True, groups of terms that are all genes or proteins, and are
            all from different species (one term from each species) are skipped.
            This is effective at eliminating ambiguities between orthologous
            genes in different species that are usually resolved using the
            organism priority list.
        """
        ambig_entries = defaultdict(list)
        for terms in self.entries.values():
            for term in terms:
                # We consider it an ambiguity if the same text entry appears
                # multiple times
                key = term.text
                ambig_entries[key].append(term)

        # It's only an ambiguity if there are two entries at least
        ambig_entries = {k: v for k, v in ambig_entries.items()
                         if len(v) >= 2}

        ambigs = []
        for text, entries in ambig_entries.items():
            dbs = {e.db for e in entries}
            db_ids = {(e.db, e.id) for e in entries}
            statuses = {e.status for e in entries}
            sources = {e.source for e in entries}
            names = {e.entry_name for e in entries}
            # If the entries all point to the same ID, we skip it
            if len(db_ids) <= 1:
                continue
            # If there is a name in statuses, we skip it because it's
            # prioritized
            if skip_names and 'name' in statuses:
                continue
            # We skip curated terms because they are prioritized anyway
            if skip_curated and 'curated' in statuses:
                continue
            # If there is an adeft model already, we skip it
            if 'adeft' in sources:
                continue
            if skip_name_matches:
                if len({e.entry_name.lower() for e in entries}) == 1:
                    continue
            if skip_species_ambigs:
                if dbs <= {'HGNC', 'UP'} and \
                        len({e.organism for e in entries}) == len(entries):
                    continue
            # Everything else is an ambiguity
            ambigs.append(entries)
        return ambigs

    def _iter_terms(self):
        for terms in self.entries.values():
            yield from terms

    def summary_str(self) -> str:
        """Summarize the contents of the grounder."""
        namespaces = {ns for term in self._iter_terms() for ns in term.get_namespaces()}
        status_counter = dict(Counter(term.status for term in self._iter_terms()))
        return dedent(f"""\
        Lookups: {len(self.entries):,}
        Terms: {sum(len(terms) for terms in self.entries.values()):,}
        Term Namespaces: {namespaces}
        Term Statuses: {status_counter}
        Adeft Disambiguators: {len(self.adeft_disambiguators):,}
        Gilda Disambiguators: {len(self.gilda_disambiguators):,}
        """)

    def print_summary(self, **kwargs) -> None:
        """Print the summary of this grounder."""
        print(self.summary_str(), **kwargs)


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
    subsumed_terms : Optional[list[gilda.grounder.Term]]
        A list of additional Term objects that also matched, have the same
        db/id value as the term associated with the match, but were further
        down the score ranking. In some cases examining the subsumed terms
        associated with a match can provide additional metadata in
        downstream applications.
    """
    def __init__(self, term: Term, score, match: Match, disambiguation=None,
                 subsumed_terms=None):
        self.term = term
        self.url = term.get_idenfiers_url()
        self.score = score
        self.match = match
        self.disambiguation = disambiguation
        self.subsumed_terms = subsumed_terms if subsumed_terms else None

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
            'url': self.url,
            'score': self.score,
            'match': self.match.to_json()
        }
        if self.disambiguation is not None:
            js['disambiguation'] = self.disambiguation
        if self.subsumed_terms:
            js['subsumed_terms'] = [term.to_json()
                                    for term in self.subsumed_terms]
        return js

    def multiply(self, value):
        logger.debug('Multiplying the score of "%s" with %.3f'
                     % (self.term.entry_name, value))
        self.score = self.score * value

    def get_namespaces(self) -> Set[str]:
        """Return all namespaces for this match including from mapped and
        subsumed terms.

        Returns
        -------
        :
            A set of strings representing namespaces for terms involved in
            this match, including the namespace for the primary term as well
            as any subsumed terms, and groundings that come from having
            mapped an original source grounding during grounding resource
            construction.
        """
        return {ns for ns, _ in self.get_groundings()}

    def get_groundings(self) -> Set[Tuple[str, str]]:
        """Return all groundings for this match including from mapped and
        subsumed terms.

        Returns
        -------
        :
            A set of tuples representing groundings for this match including
            the grounding for the primary term as well as any subsumed
            terms, and groundings that come from having mapped an original
            source grounding during grounding resource construction.
        """
        term_groundings = self.term.get_groundings()
        if self.subsumed_terms:
            for sub_term in self.subsumed_terms:
                term_groundings |= sub_term.get_groundings()
        return term_groundings

    def get_grounding_dict(self) -> Mapping[str, str]:
        """Get the groundings as CURIEs and URLs."""
        return {
            get_identifiers_curie(db, db_id): get_identifiers_url(db, db_id)
            for db, db_id in self.get_groundings()
        }


def load_entries_from_terms_file(terms_file: Union[str, Path]) -> Iterator[Term]:
    """Yield Terms from a compressed terms TSV file path.

    Parameters
    ----------
    terms_file :
        Path to a compressed TSV terms file with columns corresponding to the
        serialized elements of a Term.

    Returns
    -------
    :
        Terms loaded from the file yielded by a generator.
    """
    with gzip.open(terms_file, 'rt', encoding='utf-8') as fh:
        entries = {}
        reader = csv.reader(fh, delimiter='\t')
        # Skip header
        next(reader)
        for row in reader:
            row_nones = [r if r else None for r in row]
            yield Term(*row_nones)


def load_terms_file(terms_file: Union[str, Path]) -> Mapping[str, List[Term]]:
    """Load a TSV file containing terms into a lookup dictionary.

    Parameters
    ----------
    terms_file :
        Path to a compressed TSV terms file with columns corresponding to the
        serialized elements of a Term.

    Returns
    -------
    :
        A lookup dictionary whose keys are normalized entity texts, and values
        are lists of Terms with that normalized entity text.
    """
    entries = {}
    for term in load_entries_from_terms_file(terms_file):
        if term.norm_text in entries:
            entries[term.norm_text].append(term)
        else:
            entries[term.norm_text] = [term]
    return entries


def filter_for_organism(terms, organisms):
    # First we organize terms by organism, including None
    terms_by_organism = defaultdict(list)
    for term in terms:
        # We filter out any organisms that aren't in the list provided
        if term.organism is not None and term.organism not in organisms:
            continue
        terms_by_organism[term.organism].append(term)
    # We first take the terms without organism
    all_terms = terms_by_organism[None]
    # We now find the top organism for which we have at least
    # one term and then add the corresponding terms to the list
    # of all terms
    if set(terms_by_organism) != {None}:
        top_organism = min(set(terms_by_organism) - {None},
                           key=lambda x: organisms.index(x))
        all_terms += terms_by_organism[top_organism]
    return all_terms


def find_adeft_models():
    adeft_disambiguators = {}
    for shortform in available_adeft_models:
        adeft_disambiguators[shortform] = None
    return adeft_disambiguators


def load_adeft_models():
    return {shortform: load_disambiguator(shortform)
            for shortform in find_adeft_models()}


def load_gilda_models(cutoff=0.7):
    with gzip.open(get_gilda_models(), 'rt') as fh:
        models = {k: load_model_info(v)
                  for k, v in json.loads(fh.read()).items()
                  if v['stats']['f1']['mean'] > cutoff}
    return models
