from gilda.process import depluralize, replace_greek_spelled_out


def test_depluralize():
    assert depluralize('BRAF') == ('BRAF', 'non_plural')
    assert depluralize('apoptosis') == ('apoptosis', 'non_plural')
    assert depluralize('mosquitoes') == ('mosquito', 'plural_oes')
    assert depluralize('antibodies') == ('antibody', 'plural_ies')
    assert depluralize('branches') == ('branch', 'plural_es')
    assert depluralize('CDs') == ('CD', 'plural_caps_s')
    assert depluralize('receptors') == ('receptor', 'plural_s')


def test_greek():
    assert replace_greek_spelled_out('interferon-γ') == \
        'interferon-gamma'
