__all__ = ['ground', 'get_models', 'get_names', 'get_grounder', 'make_grounder']

from typing import List, Mapping, Union, Optional

from gilda.grounder import Grounder, Term


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
    """
    return grounder.ground(text=text, context=context, organisms=organisms)


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
        terms: Union[str, List[Term], Mapping[str, List[Term]]]) -> Grounder:
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
    """
    return Grounder(terms=terms)
