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


def test_parse_embedded_parentheses_uniprot_2():
    txt = ('Na(+)/H(+) exchange regulatory cofactor NHE-RF1 (NHERF-1) '
           '(Ezrin-radixin-moesin-binding phosphoprotein 50) (EBP50)'
           ' (Regulatory cofactor of Na(+)/H(+) exchanger) '
           '(Sodium-hydrogen exchanger regulatory factor 1) '
           '(Solute carrier family 9 isoform A3 regulatory factor 1)')
    syms = parse_uniprot_synonyms(txt)
    for s in syms:
        assert not s == '+/', '"%s"' % s
    assert len(syms) == 7, syms
    assert 'Na(+)/H(+) exchange regulatory cofactor NHE-RF1' in syms, syms


def test_parse_embedded_parentheses_uniprot_3():
    txt = ('Solute carrier family 13 member 3 (Na(+)/dicarboxylate ' \
           'cotransporter 3, NaDC-3, rNaDC3) (Sodium-dependent high-affinity' \
           ' dicarboxylate transporter 2)')
    syms = parse_uniprot_synonyms(txt)
    assert syms == \
        ['Solute carrier family 13 member 3',
         'Na(+)/dicarboxylate cotransporter 3, NaDC-3, rNaDC3',
         'Sodium-dependent high-affinity dicarboxylate transporter 2']


def test_parse_parentheses_in_name():
    txt = 'DNA (cytosine-5)-methyltransferase 1 (EC:2.1.1.37)'
    syms = parse_uniprot_synonyms(txt)
    assert syms == ['DNA (cytosine-5)-methyltransferase 1', 'EC:2.1.1.37'], \
        syms

