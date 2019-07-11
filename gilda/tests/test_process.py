from gilda.process import depluralize


def test_depluralize():
    assert depluralize('BRAF') == ('BRAF', 'non_plural')
    assert depluralize('apoptosis') == ('apoptosis', 'non_plural')
    assert depluralize('mosquitoes') == ('mosquito', 'plural_oes')
    assert depluralize('antibodies') == ('antibody', 'plural_ies')
    assert depluralize('branches') == ('branch', 'plural_es')
    assert depluralize('CDs') == ('CD', 'plural_caps_s')
    assert depluralize('receptors') == ('receptor', 'plural_s')