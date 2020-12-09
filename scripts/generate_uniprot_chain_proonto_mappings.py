import csv
import obonet
from collections import defaultdict
from gilda.term import Term
from gilda.grounder import Grounder, normalize

# This is from the Reach bioresources file as a convenient
# source of pre-processed synonyms
UNIPROT = '/Users/ben/src/bioresources/kb/uniprot-proteins.tsv'
PROONTO = '/Users/ben/src/bioresources/kb/protein-ontology-fragments.tsv'
PROONTO_OBO = '/Users/ben/src/bioresorces/scripts/pro_reasoned.obo'


if __name__ == '__main__':
    # 1. Parse all the UniProt synonyms that are for human
    # protein fragments into Gilda Terms
    terms = []
    with open(UNIPROT, 'r') as fh:
        reader = csv.reader(fh, delimiter='\t')
        for row in reader:
            if '#' in row[1] and row[2] == 'Human':
                synonym = row[0]
                id = row[1].split('#')[1]
                term = Term(normalize(synonym), synonym, 'UP',
                            id, synonym, 'synonym', 'uniprot')
                terms.append(term)
    # 2. Instantiate a grounder with these terms
    terms_dict = defaultdict(list)
    pro = obonet.read_obo(PROONTO_OBO)
    for term in terms:
        terms_dict[term.norm_text].append(term)
    grounder = Grounder(terms_dict)
    # 3. Parse all the Protein Ontology synonyms and ground each of them, then
    # store the results
    matches_per_id = defaultdict(list)
    with open(PROONTO, 'r') as fh:
        reader = csv.reader(fh, delimiter='\t')
        for row in reader:
            synonym, id = row
            matches = grounder.ground(synonym)
            if matches:
                matches_per_id[id] += matches
    # 4. Dump spreadsheet with non-ambiguous equivalences in BioMappings formas