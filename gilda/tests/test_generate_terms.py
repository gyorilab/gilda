from gilda.generate_terms import parse_uniprot_synonyms


def test_parse_embedded_parentheses_uniprot():
    txt = ('Sodium-coupled neutral amino acid transporter 4 (Amino acid '
           'transporter A3) (Na(+)-coupled neutral amino acid transporter 4)'
           ' (Solute carrier family 38 member 4) (System A amino acid '
           'transporter 3)')
    syms = parse_uniprot_synonyms(txt)
    for s in syms:
        assert not s.endswith(' '), '"%s"' % s
    assert len(syms) == 5, syms
    assert 'System A amino acid transporter 3' in syms