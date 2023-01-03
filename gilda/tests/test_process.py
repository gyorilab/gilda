from gilda.process import depluralize, replace_greek_spelled_out, \
    replace_roman_arabic, replace_greek_uni, replace_greek_latin, normalize


def test_depluralize():
    assert depluralize('BRAF') == [('BRAF', 'non_plural')]
    assert depluralize('apoptosis') == [('apoptosis', 'non_plural')]
    assert depluralize('mosquitoes') == [('mosquito', 'plural_oes'),
                                         ('mosquitoe', 'plural_s')]
    assert depluralize('antibodies') == [('antibody', 'plural_ies'),
                                         ('antibodie', 'plural_s')]
    assert depluralize('branches') == [('branch', 'plural_es'),
                                       ('branche', 'plural_s')]
    assert depluralize('CDs') == [('CD', 'plural_caps_s')]
    assert depluralize('receptors') == [('receptor', 'plural_s')]
    assert depluralize('kinases') == [('kinas', 'plural_es'),
                                      ('kinase', 'plural_s')]


def test_greek():
    assert replace_greek_spelled_out('interferon-γ') == 'interferon-gamma'
    assert replace_greek_uni('interferon-gamma') == 'interferon-γ'
    assert replace_greek_latin('interferon-beta') == 'interferon-b'


def test_roman_arabic():
    assert replace_roman_arabic('xx-1') == 'xx-I'
    assert replace_roman_arabic('xx-10') == 'xx-X'
    assert replace_roman_arabic('x1x') == 'x1x'
    assert replace_roman_arabic('xx-I') == 'xx-1'
    assert replace_roman_arabic('xx viii') == 'xx 8'
    assert replace_roman_arabic('xx-iX') == 'xx-9'


def test_normalize():
    assert normalize('Löfgren’s syndrome') == 'lofgren\'s syndrome'
    assert normalize('βAR') == 'βar'