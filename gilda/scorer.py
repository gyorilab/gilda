from .process import replace_dashes, replace_whitespace, normalize


class Match(object):
    def __init__(self, query, ref, exact):
        self.query = query
        self.ref = ref
        self.exact = exact


def generate_match(query, ref, beginning_of_sentence):
    """Return a match data structure based on comparing a query to a ref str.

    Parameters
    ----------
    query : str
        The string to be compared against a reference string.
    ref : str
        The reference string against which the incoming query string is
        compared.
    beginning_of_sentence : bool
        True if the query_str appears at the beginning of a sentence, relevant
        for how capitalization is evaluated.

    Returns
    -------
    dict
        A dictionary characterizing the match between the two strings.
    """
    # First of all, this function assumes that both the query and the ref have
    # been normalized and so if that is not the case, we raise an error
    if normalize(query) != normalize(ref):
        raise ValueError('Normalized query "%s" does not match normalized'
                         ' reference "%s"' % (query, ref))

    # Pre-process both strings first by replacing multiple white spaces
    # with a single ASCII space, and all kinds of dashes with a single
    # ASCII dash.
    query = replace_dashes(replace_whitespace(query))
    ref = replace_dashes(replace_whitespace(ref))

    # If we have an exact match at this point then we can return immediately
    if not beginning_of_sentence and query == ref:
        return Match(query, ref, exact=True)

    query_pieces = re.split('')
