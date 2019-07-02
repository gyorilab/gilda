"""Module containing various string processing functions used for grounding."""
import regex as re
from .greek_alphabet import greek_alphabet, greek_to_latin


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


def replace_greek_uni(s):
    """Replace Greek spelled out letters with their unicode character."""
    for greek_uni, greek_spelled_out in greek_alphabet.items():
        s = s.replace(greek_spelled_out, greek_uni)
    return s


def replace_greek_latin(s):
    """Replace Greek spelled out letters with their latin character."""
    for greek_spelled_out, latin in greek_to_latin.items():
        s = s.replace(greek_spelled_out, latin)
    return s


def get_capitalization_pattern(word, beginning_of_sentence=False):
    """Return the type of capitalization for the string.

    Parameters
    ----------
    word : str
        The word whose capitalization is determined.
    beginning_of_sentence : Optional[bool]
        True if the word appears at the beginning of a sentence. Default: False

    Returns
    -------
    str
        The capitalization pattern of the given word. Returns one of the
        following: sentence_initial_cap, single_cap_letter, all_caps, all_lower,
        initial_cap, mixed.

    """
    if beginning_of_sentence and re.match(r'^\p{Lu}\p{Ll}*$', word):
        return 'sentence_initial_cap'
    elif re.match(r'^\p{Lu}$', word):
        return 'single_cap_letter'
    elif re.match(r'^\p{Lu}+$', word):
        return 'all_caps'
    elif re.match(r'^\p{Ll}+$', word):
        return 'all_lower'
    elif re.match(r'^\p{Lu}\p{Ll}+$', word):
        return 'initial_cap'
    else:
        return 'mixed'
