"""Module containing various string processing functions used for grounding."""
from typing import List, Tuple

import regex as re
import unidecode

from .greek_alphabet import greek_alphabet, greek_to_latin


#: A list of all kinds of dashes
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
    s = replace_unicode(s)
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


def replace_greek_spelled_out(s):
    """Replace Greek unicode character with latin spelled out.
    """
    for greek_uni, greek_spelled_out in greek_alphabet.items():
        s = s.replace(greek_uni, greek_spelled_out)
    return s


def replace_unicode(s):
    """Replace unicode with ASCII equivalent, except Greek letters.

    Greek letters are handled separately and aren't replaced in this context.
    """
    if unidecode.unidecode(s) == s:
        return s
    return ''.join(unidecode.unidecode(c) if c not in greek_alphabet else c
                   for c in s)


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


def depluralize(word: str) -> List[Tuple[str, str]]:
    """Return the depluralized version of the word, along with a status flag.

    Parameters
    ----------
    word : str
        The word which is to be depluralized.

    Returns
    -------
    list of str pairs:
        The original word, if it is detected to be non-plural, or the
        depluralized version of the word, and a status flag representing the
        detected pluralization status of the
        word, with non_plural (e.g., BRAF), plural_oes (e.g., mosquitoes),
        plural_ies (e.g., antibodies), plural_es (e.g., switches),
        plural_cap_s (e.g., MAPKs), and plural_s (e.g., receptors).
    """
    # If the word doesn't end in s, we assume it's not plural
    if not word.endswith('s'):
        return [(word, 'non_plural')]
    # Another case is words ending in -sis (e.g., apoptosis), these are almost
    # exclusively non plural so we return here too
    elif word.endswith('sis'):
        return [(word, 'non_plural')]
    # This is the case when the word ends with an o which is pluralized as oes
    # e.g., mosquitoes
    elif word.endswith('oes'):
        return [(word[:-2], 'plural_oes'),
                (word[:-1], 'plural_s')]
    # This is the case when the word ends with a y which is pluralized as ies,
    # e.g., antibodies
    elif word.endswith('ies'):
        return [(word[:-3] + 'y', 'plural_ies'),
                (word[:-1], 'plural_s')]
    # These are the cases where words form plurals by adding -es so we
    # return by stripping it off. However, it's not possible to determine
    # if the word doesn't end in e.g., -xe or -se in a singluar form, and
    # so we also return a variant to account for this.
    elif word.endswith(('xes', 'ses', 'ches', 'shes')):
        return [(word[:-2], 'plural_es'), (word[:-1], 'plural_s')]
    # If the word is all caps and the last letter is an s, then it's a very
    # strong signal that it is pluralized so we have a custom return value
    # for that
    elif re.match(r'^\p{Lu}+$', word[:-1]):
        return [(word[:-1], 'plural_caps_s')]
    # Otherwise, we just go with the assumption that the last s is the
    # plural marker
    else:
        return [(word[:-1], 'plural_s')]
    # Note: there don't seem to be any compelling examples of -f or -fe -> ves
    # so it is not implemented


def replace_roman_arabic(s):
    match = roman_arabic_prefilter.match(s)
    if not match:
        return s
    else:
        pattern = roman_arabic_patterns.get(match.groups()[0].upper())
        return pattern[0].sub(pattern[1], s) if pattern else s


def _make_roman_arabic_patterns():
    roman_arabic = {
        'I': '1',
        'II': '2',
        'III': '3',
        'IV': '4',
        'V': '5',
        'VI': '6',
        'VII': '7',
        'VIII': '8',
        'IX': '9',
        'X': '10'
    }

    roman_arabic_patterns = {}
    for r, a in roman_arabic.items():
        for a, b in [(r, a), (a, r)]:
            roman_arabic_patterns[a] = (re.compile(r'^(.*[- ])(%s)$' % a,
                                                  re.IGNORECASE),
                                        r'\g<1>%s' % b)
    return roman_arabic_patterns


roman_arabic_patterns = _make_roman_arabic_patterns()
roman_arabic_prefilter = re.compile(r'^.*[- ](\d+|[IXV]+)$', re.IGNORECASE)
