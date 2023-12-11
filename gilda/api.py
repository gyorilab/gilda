__all__ = [
    'ground',
    'get_models',
    'get_names',
    'get_grounder',
    'make_grounder',
    'annotate',
]

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

    @property
    def prefix_index(self):
        return self.get_grounder().prefix_index


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

    The rationale for each match is contained in the ``match`` attribute
    whose fields are described in :class:`gilda.scorer.Match`:

    >>> match_object = scored_matches[0].match

    Give optional context to be used by Gilda's disambiguation models, if available

    >>> scored_matches = gilda.ground('ER', context='Calcium is released from the ER.')

    Only return results from a certain namespace, such as when a family and gene have the same name

    >>> scored_matches = gilda.ground('ESR', namespaces=["hgnc"])
    """
    return grounder.ground(text=text, context=context, organisms=organisms, namespaces=namespaces)


def annotate(
    text: str,
    sent_split_fun=None,
    organisms=None,
    namespaces=None,
    return_first: bool = True,
):
    """Annotate a given text with Gilda (i.e., do named entity recognition).

    Parameters
    ----------
    text : str
        The text to be annotated.
    sent_split_fun : Callable, optional
        A function that splits the text into sentences. The default is
        :func:`nltk.tokenize.sent_tokenize`. The function should take a string
        as input and return an iterable of strings corresponding to the sentences
        in the input text.
    organisms : list[str], optional
        A list of organism names to pass to the grounder. If not provided,
        human is used.
    namespaces : list[str], optional
        A list of namespaces to pass to the grounder to restrict the matches
        to. By default, no restriction is applied.
    return_first:
        If true, only returns the first result. Otherwise, returns all results.

    Returns
    -------
    list[tuple[str, ScoredMatch, int, int]]
        A list of tuples of start and end character offsets of the text
        corresponding to the entity, the entity text, and the ScoredMatch
        object corresponding to the entity.
    """
    from .ner import annotate as _annotate

    return _annotate(
        text,
        grounder=grounder,
        sent_split_fun=sent_split_fun,
        organisms=organisms,
        namespaces=namespaces,
        return_first=return_first,
    )


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
    The following example shows how to get an ontology with :mod:`obonet` and
    load custom terms:

    .. code-block:: python

        from gilda import make_grounder
        from gilda.process import normalize
        from gilda import Term

        prefix = "UBERON"
        url = "http://purl.obolibrary.org/obo/uberon/basic.obo"
        g = obonet.read_obo(url)
        custom_obo_terms = []
        it = tqdm(g.nodes(data=True), unit_scale=True, unit="node")
        for node, data in it:
            # Skip entries imported from other ontologies
            if not node.startswith(f"{prefix}:"):
                continue

            identifier = node.removeprefix(f"{prefix}:")

            name = data["name"]
            custom_obo_terms.append(gilda.Term(
                norm_text=normalize(name),
                text=name,
                db=prefix,
                id=identifier,
                entry_name=name,
                status="name",
                source=prefix,
            ))

            # Add terms for all synonyms
            for synonym_raw in data.get("synonym", []):
                try:
                    # Try to parse out of the quoted OBO Field
                    synonym = synonym_raw.split('"')[1].strip()
                except IndexError:
                    continue  # the synonym was malformed

                custom_obo_terms.append(gilda.Term(
                    norm_text=normalize(synonym),
                    text=synonym,
                    db=prefix,
                    id=identifier,
                    entry_name=name,
                    status="synonym",
                    source=prefix,
                ))

        custom_grounder = gilda.make_grounder(custom_obo_terms)
        scored_matches = custom_grounder.ground("head")

    Additional examples for loading custom content from OBO Graph JSON,
    :mod:`pyobo`, and more can be found in the `Jupyter notebooks
    <https://github.com/indralab/gilda/tree/master/notebooks>`_
    in the Gilda repository on GitHub.
    """
    return Grounder(terms=terms)
