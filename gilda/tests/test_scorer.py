from gilda.process import get_capitalization_pattern
from gilda.scorer import generate_match, score_string_match
from . import appreq


def test_cap_pattern():
    assert get_capitalization_pattern('Abcd', True) == 'sentence_initial_cap'
    assert get_capitalization_pattern('A') == 'single_cap_letter'
    assert get_capitalization_pattern('BRAF') == 'all_caps'
    assert get_capitalization_pattern('abcd') == 'all_lower'
    assert get_capitalization_pattern('Abcd') == 'initial_cap'
    assert get_capitalization_pattern('aBcd') == 'mixed'


def test_generate_match():
    match = generate_match('BRAF', 'Braf')
    assert ('all_caps', 'initial_cap') in match.cap_combos, match

    match = generate_match('Braf', 'BRAF')
    assert ('initial_cap', 'all_caps') in match.cap_combos, match

    match = generate_match('BrAf', 'braf')
    assert ('mixed', 'all_lower') in match.cap_combos, match


def test_characterize_match_dashes():
    match = generate_match('eIF-5A', 'EIF5A')
    assert 'query' in match.dash_mismatches
    assert ('mixed', 'all_caps') in match.cap_combos


def test_string_match_scoring():
    cases = [
        ('k-ras', 'K-ras', 0.9443),
        ('k-ras', 'KRAS', 0.8589),
        ('k-ras', 'Kras', 0.9424),
        ('ras', 'RAs', 0.6938),
        ('ras', 'ras', 1),
        ('Ras', 'RA-s', 0.9425),
        ('ras', 'RAS', 0.3599)
    ]
    for query, ref, expected_score in cases:
        match = generate_match(query, ref)
        score = score_string_match(match)
        assert appreq(score, expected_score), (score, expected_score)

