from gilda.term import Term
from gilda.generate_terms import parse_uniprot_synonyms, \
    filter_out_duplicates, get_terms_from_uniprot_row


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


def test_parse_parantheses_in_name2():
    txt = ('Neutral amino acid transporter B(0) (ATB(0)) '
           '(Baboon M7 virus receptor) (RD114/simian type D retrovirus '
           'receptor) (Sodium-dependent neutral amino acid transporter type 2) '
           '(Solute carrier family 1 member 5)')
    # The challenge here is the trailing (0) at the end of the first
    # synonym which is part of that synonym and shouldn't be treated as
    # a separate synonym.
    syms = parse_uniprot_synonyms(txt)
    assert syms == \
        ['Neutral amino acid transporter B(0)',
         'ATB(0)',
         'Baboon M7 virus receptor',
         'RD114/simian type D retrovirus receptor',
         'Sodium-dependent neutral amino acid transporter type 2',
         'Solute carrier family 1 member 5']


def test_filter_priority():
    term1 = Term('mekk2', 'MEKK2', 'HGNC', '6854', 'MAP3K2',
                 'former_name', 'hgnc', '9606')
    term2 = Term('mekk2', 'MEKK2', 'HGNC', '6854', 'MAP3K2',
                 'synonym', 'up', '9606')
    terms = filter_out_duplicates([term1, term2])
    assert len(terms) == 1
    term = terms[0]
    assert term.status == 'synonym'


def test_filter_priority_by_source():
    term1 = Term('mekk2', 'MEKK2', 'HGNC', '6854', 'MAP3K2',
                 'synonym', 'hgnc', '9606')
    term2 = Term('mekk2', 'MEKK2', 'HGNC', '6854', 'MAP3K2',
                 'synonym', 'up', '9606')
    terms = filter_out_duplicates([term1, term2])
    assert len(terms) == 1
    assert terms[0] == term1

    # now test the other way, to make sure order doesn't matter
    terms = filter_out_duplicates([term2, term1])
    assert len(terms) == 1
    assert terms[0] == term1


def test_get_terms_simple():
    row = {'Entry': 'P15056',
           'Gene names  (primary )': 'BRAF',
           'Gene names  (synonym )': 'BRAF1 RAFB1',
           'Protein names':
               ('Serine/threonine-protein kinase B-raf '
                '(EC 2.7.11.1) (Proto-oncogene B-Raf) (p94) '
                '(v-Raf murine sarcoma viral oncogene homolog B1)'),
           'Organism ID': '9606'}
    terms = get_terms_from_uniprot_row(row)
    assert len(terms) == 7, terms
    assert all(term.db == 'HGNC' for term in terms), terms


def test_get_terms_multi_gene_human():
    row = {'Entry': 'P62805',
           'Gene names  (primary )':
               ('H4C1; H4C2; H4C3; H4C4; H4C5; H4C6; H4C8;'
                ' H4C9; H4C11; H4C12; H4C13; H4C14; H4C15; H4-16'),
           'Gene names  (synonym )': ('H4/A H4FA HIST1H4A; H4/I H4FI HIST1H4B; '
                                      'H4/G H4FG HIST1H4C; H4/B H4FB HIST1H4D; '
                                      'H4/J H4FJ HIST1H4E; H4/C H4FC HIST1H4F; '
                                      'H4/H H4FH HIST1H4H; H4/M H4FM HIST1H4I; '
                                      'H4/E H4FE HIST1H4J; H4/D H4FD HIST1H4K; '
                                      'H4/K H4FK HIST1H4L; H4/N H4F2 H4FN '
                                      'HIST2H4 HIST2H4A; H4/O H4FO HIST2H4B; '
                                      'HIST4H4'),
           'Protein names': 'Histone H4',
           'Organism ID': '9606'}
    terms = get_terms_from_uniprot_row(row)
    assert len(terms) == 1, terms
    assert terms[0].db == 'UP'
    assert terms[0].text == 'Histone H4'


def test_get_terms_multi_gene_nonhuman():
    row = {'Entry': 'P62784',
           'Gene names  (primary )': ('his-1; his-5; his-10; his-14; his-18; '
                                      'his-26; his-28; his-31; his-37; his-38; '
                                      'his-46; his-50; his-56; his-60; '
                                      'his-64; his-67'),
           'Gene names  (synonym )': '; ; ; ; ; ; ; ; ; ; ; ; ; ; ; ',
           'Protein names': 'Histone H4',
           'Organism ID': '6239'}
    terms = get_terms_from_uniprot_row(row)
    assert len(terms) == 17
