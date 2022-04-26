__all__ = ['ground', 'get_models', 'get_names', 'get_grounder', 'make_grounder']

from typing import List, Mapping, Union, Optional

from gilda.grounder import Grounder
from gilda.term import Term


class GrounderInstance(object):
    def __init__(self):
        self.grounder = None

    def get_grounder(self):
        if self.grounder is None:
            self.grounder = Grounder()
        return self.grounder

    def ground(self, text, context=None, organisms=None,
               namespaces=None):
        return self.get_grounder().ground(text, context=context,
                                          organisms=organisms,
                                          namespaces=namespaces)

    def get_models(self):
        return self.get_grounder().get_models()

    def get_names(self, db, id, status=None, source=None):
        return self.get_grounder().get_names(db, id,
                                             status=status,
                                             source=source)


grounder = GrounderInstance()


def ground(text, context=None, organisms=None, namespaces=None):
    """Return a list of scored matches for a text to ground.

    Parameters
    ----------
    text : str
        The entity text to be grounded.
    context : Optional[str]
        Any additional text that serves as context for disambiguating the
        given entity text, used if a model exists for disambiguating the
        given text.
    organisms : Optional[List[str]]
        A list of taxonomy identifiers to use as a priority list
        when surfacing matches for proteins/genes from multiple organisms.
    namespaces : Optional[List[str]]
        A list of namespaces to restrict the matches to. By default, no
        restriction is applied.

    Returns
    -------
    list[gilda.grounder.ScoredMatch]
        A list of ScoredMatch objects representing the groundings.

    Examples
    --------
    Ground a string corresponding to an entity name, label, or synonym

    >>> import gilda
    >>> scored_matches = gilda.ground('mapt')

    The matches are sorted in descending order by score, and in the event of
    a tie, by the namespace of the primary grounding. Each scored match has a
    :class:`gilda.term.Term` object that contain information about the primary
    grounding.

    >>> scored_matches[0].term.db
    'hgnc'
    >>> scored_matches[0].term.id
    '6893'
    >>> scored_matches[0].term.get_curie()
    'hgnc:6893'

    The score for each match can be accessed directly:

    >>> scored_matches[0].score
    0.7623

    The rational for each match is contained in the ``match`` attribute
    whose fields are described in :class:`gilda.scorer.Match`:

    >>> match_object = scored_matches[0].match

    Give optional context to be used by Gilda's disambiguation models, if available

    >>> scored_matches = gilda.ground('ER', context='Calcium is released from the ER.')

    Only return results from a certain namespace, such as when a family and gene have the same name

    >>> scored_matches = gilda.ground('ESR', namespaces=["hgnc"])
    """
    return grounder.ground(text=text, context=context, organisms=organisms, namespaces=namespaces)


def get_models():
    """Return a list of entity texts for which disambiguation models exist.

    Returns
    -------
    list[str]
        The list of entity texts for which a disambiguation model is
        available.
    """
    return grounder.get_models()


def get_names(db, id, status=None, source=None):
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
    """
    return grounder.get_names(db, id, status=status, source=source)


def get_grounder() -> Grounder:
    """Initialize and return the default Grounder instance.

    Returns
    -------
    :
        A Grounder instance whose attributes and methods can be used
        directly.
    """
    return grounder.get_grounder()


def make_grounder(
    terms: Union[str, List[Term], Mapping[str, List[Term]]],
) -> Grounder:
    """Create a custom grounder from a list of Terms.

    Parameters
    ----------
    terms :
        Specifies the grounding terms that should be loaded in the Grounder.
        If str, it is interpreted as a path to a grounding
        terms gzipped TSV file which is then loaded. If list, it is assumed to
        be a flat list of Terms. If dict, it is assumed to be a grounding terms
        dict with normalized entity strings as keys and lists of Term objects
        as values.
        Default: None

    Returns
    -------
    :
        A Grounder instance, initialized with either the default terms
        loaded from the resource file or a custom set of terms
        if the terms argument was specified.

    Examples
    --------
    While the NCBITaxon namespace is often used as the gold standard for species,
    there are several complementary other species- and taxonomy-centric
    namespaces, such as the `Integrated Taxonomic Information System
    <https://bioregistry.io/registry/itis>`_. These can be loaded in Gilda via
    :mod:`pyobo` like in the following example
    (note, sometimes ITIS is unresponsive):

    .. code-block:: python

        import gilda, pyobo
        from gilda.process import normalize
        custom_species_terms: list[gilda.Term] = []
        itis_names = pyobo.get_id_name_mapping("itis")
        itis_synonyms = pyobo.get_id_synonyms_mapping("itis")
        for identifier, name in itis_names.items():
            custom_species_terms.append(gilda.Term(
                norm_text=normalize(name),
                text=name,
                db="itis",
                id=identifier,
                entry_name=name,
                status="name",
                source="itis",
            ))
            for synonym in itis_synonyms.get(identifier, []):
                custom_species_terms.append(gilda.Term(
                    norm_text=normalize(synonym),
                    text=synonym,
                    db="itis",
                    id=identifier,
                    entry_name=name,
                    status="synonym",
                    source="itis",
                ))
        custom_grounder = gilda.make_grounder(custom_species_terms)
        custom_grounder.ground("e coli")

    Similarly, PyOBO can be used to generate a grounder containing
    multiple pathway databases' names

    .. code-block:: python

        import gilda, pyobo
        from gilda.process import normalize

        custom_pathway_terms = []
        for prefix in ["reactome", "wikipathways", "pw"]:
            names = pyobo.get_id_name_mapping(prefix)
            synonyms = pyobo.get_id_synonyms_mapping(prefix)
            custom_pathway_terms.append(gilda.Term(
                norm_text=normalize(name),
                text=name,
                db=prefix,
                id=identifier,
                entry_name=name,
                status="name",
                source=prefix,
            ))
            for synonym in synonyms.get(identifier, []):
                custom_pathway_terms.append(gilda.Term(
                    norm_text=normalize(synonym),
                    text=synonym,
                    db=prefix,
                    id=identifier,
                    entry_name=name,
                    status="synonym",
                    source=prefix,
                ))
        custom_pathway_grounder = gilda.make_grounder(custom_species_terms)
        custom_pathway_grounder.ground("apoptosis")
    """
    return Grounder(terms=terms)
