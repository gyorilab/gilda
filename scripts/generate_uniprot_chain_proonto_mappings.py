import csv
import obonet
from collections import defaultdict
from gilda.term import Term
from gilda.grounder import Grounder, normalize

# This is from the Reach bioresources file as a convenient
# source of pre-processed synonyms
UNIPROT = '/Users/ben/src/bioresources/kb/uniprot-proteins.tsv'
PROONTO = '/Users/ben/src/bioresources/kb/protein-ontology-fragments.tsv'
PROONTO_OBO = '/Users/ben/src/bioresources/scripts/pro_reasoned.obo'
BIOMAPPINGS = '/Users/ben/Dropbox/postdoc/darpa/src/biomappings/'\
    'predictions.tsv'


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
    # 4. Dump spreadsheet with non-ambiguous equivalences in BioMappings format
    # source prefix, source identifier, source name, relation
    # target prefix, target identifier, target name, type, source
    source_prefix = 'pr'
    target_prefix = 'uniprot.chain'
    relation = 'skos:exactMatch'
    source = 'https://github.com/indralab/gilda/blob/master/scripts/' \
        'generate_uniprot_chain_proonto_mappings.py'
    match_type = 'lexical'
    rows = []
    for pro_id, matches in matches_per_id.items():
        if len(matches) > 1:
            print(matches)
            continue
        target_id = matches[0].term.id
        target_name = matches[0].term.entry_name
        source_name = pro.nodes[pro_id]['name']
        row = [source_prefix, pro_id, source_name, relation,
               target_prefix, target_id, target_name, match_type,
               source]
        rows.append(row)
    with open(BIOMAPPINGS, 'a') as fh:
        fh.write('\n'.join(['\t'.join(row) for row in rows]))