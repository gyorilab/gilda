__all__ = ['ground', 'get_models']

from gilda.grounder import Grounder
from gilda.resources import get_grounding_terms


class GrounderInstance(object):
    def __init__(self):
        self.grounder = None

    def get_grounder(self):
        if self.grounder is None:
            self.grounder = Grounder(get_grounding_terms())
        return self.grounder


grounder = GrounderInstance()


def ground(text, context=None):
    """Return a list of scored matches for a text to ground.

    Parameters
    ----------
    text : str
        The entity text to be grounded.
    context : Optional[str]
        Any additional text that serves as context for disambiguating the
        given entity text, used if a model exists for disambiguating the
        given text.

    Returns
    -------
    list[gilda.grounder.ScoredMatch]
        A list of ScoredMatch objects representing the groundings.
    """
    scored_matches = grounder.get_grounder().ground(text)
    if context:
        scored_matches = grounder.get_grounder().disambiguate(text,
                                                              scored_matches,
                                                              context)
    scored_matches = sorted(scored_matches, key=lambda x: x.score,
                            reverse=True)
    return scored_matches


def get_models():
    """Return a list of entity texts for which disambiguation models exist.

    Returns
    -------
    list[str]
        The list of entity texts for which a disambiguation model is
        available.
    """
    return sorted(list(grounder.get_grounder().gilda_disambiguators.keys()))
