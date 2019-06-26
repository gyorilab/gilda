"""Module containing various string processing functions used for grounding."""
import regex as re


# We try to list all kinds of dashes here
dashes = [chr(0x2212), chr(0x002d)] + [chr(c) for c in range(0x2010, 0x2016)]


def replace_dashes(s, rep='-'):
    """Replace all types of dashes in a given string with a given replacement.

    Parameters
    ----------
    s : str
        The string in which all types of dashes should be replaced.
    rep : Optional[str]
        The string with which dashes should be replaced. By default, the plain
        ASCII dash (-) is used.

    Returns
    -------
    str
        The string in which dashes have been replaced.
    """
    for d in dashes:
        s = s.replace(d, rep)
    return s


def remove_dashes(s):
    """Remove all types of dashes in the given string.

    Parameters
    ----------
    s : str
        The string in which all types of dashes should be replaced.

    Returns
    -------
    str
        The string from which dashes have been removed.
    """
    return replace_dashes(s, '')


def replace_whitespace(s, rep=' '):
    """Replace any length white spaces in the given string with a replacement.

    Parameters
    ----------
    s : str
        The string in which any length whitespaces should be replaced.
    rep : Optional[str]
        The string with which all whitespace should be replaced. By default,
        the plain ASCII space ( ) is used.

    Returns
    -------
    str
        The string in which whitespaces have been replaced.
    """
    s = re.sub(r'\s+', rep, s)
    return s


def normalize(s):
    """Normalize white spaces, dashes and case of a given string.

    Parameters
    ----------
    s : str
        The string to be normalized.

    Returns
    -------
    str
        The normalized string.
    """
    s = replace_whitespace(s)
    s = remove_dashes(s)
    s = s.lower()
    return s


def split_preserve_tokens(s):
    """Return split words of a string including the non-word tokens.

    Parameters
    ----------
    s : str
        The string to be split.

    Returns
    -------
    list of str
        The list of words in the string including the separator tokens,
        typically spaces and dashes..
    """
    return re.split(r'(\W)', s)
