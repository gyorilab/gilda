from gilda import Term, Grounder

# This is from the Reach bioresources file as a convenient
# source of pre-processed synonyms
UNIPROT = '/Users/ben/src/bioresources/kb/uniprot-proteins.tsv'
PROONTO = '/Users/ben/src/bioresources/kb/protein-ontology-fragments.tsv'


# 1. Parse all the UniProt synonyms that are for human
# protein fragments into Gilda Terms
# 2. Instantiate a grounder with these terms
# 3. Parse all the Protein Ontology synonyms and ground each of them, then
# store the results
# 4. Dump spreadsheet with non-ambiguous equivalences in BioMappings format
