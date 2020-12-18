import re
import csv
import obonet
from collections import defaultdict
from gilda.term import Term
from gilda.grounder import Grounder, normalize
from biomappings.resources import append_prediction_tuples

# This is from the Reach bioresources file as a convenient
# source of pre-processed synonyms
UNIPROT = '/Users/ben/src/bioresources/kb/uniprot-proteins.tsv'
PROONTO = '/Users/ben/src/bioresources/kb/protein-ontology-fragments.tsv'
PROONTO_OBO = '/Users/ben/src/bioresources/scripts/pro_reasoned.obo'


def organism_filter(organism):
    if organism == 'Human':
        return True
    if organism == 'SARS-CoV-2':
        return True
    return False


def get_uniprot_terms():
    terms = {}
    with open(UNIPROT, 'r') as fh:
        reader = csv.reader(fh, delimiter='\t')
        for row in reader:
            if '#' in row[1] and organism_filter(row[2]):
                synonym = row[0]
                id = row[1].split('#')[1]
                term = Term(normalize(synonym), synonym, 'UP',
                            id, synonym, 'synonym', 'uniprot')
                terms[str(term.to_json())] = term
    return list(terms.values())


def pre_process_synonym(synonym):
    synonyms = []
    synonyms.append(synonym)
    remove_suffix = re.sub(r'(.*) (\(.*)\)$', '\\1', synonym)
    synonyms.append(remove_suffix)
    return synonyms


def ground_proonto_terms(grounder):
    matches_per_id = defaultdict(list)
    with open(PROONTO, 'r') as fh:
        reader = csv.reader(fh, delimiter='\t')
        for row in reader:
            synonym, id = row
            for text in pre_process_synonym(synonym):
                matches = grounder.ground(text)
                if matches:
                    matches_per_id[id] += matches
    return matches_per_id


def dump_predictions():
    # source prefix, source identifier, source name, relation
    # target prefix, target identifier, target name, type, source
    source_prefix = 'pr'
    target_prefix = 'uniprot.chain'
    relation = 'skos:exactMatch'
    source = 'https://github.com/indralab/gilda/blob/master/scripts/' \
        'generate_uniprot_chain_proonto_mappings.py'
    match_type = 'lexical'
    rows = []
    pro = obonet.read_obo(PROONTO_OBO)
    for pro_id, matches in matches_per_id.items():
        target_id = matches[0].term.id
        target_name = matches[0].term.entry_name
        source_name = pro.nodes[pro_id]['name']
        row = (source_prefix, pro_id, source_name, relation,
               target_prefix, target_id, target_name, match_type,
               0.8, source)
        rows.append(row)
    append_prediction_tuples(rows, deduplicate=True)


if __name__ == '__main__':
    # 1. Parse all the UniProt synonyms that are for human
    # protein fragments into Gilda Terms
    terms = get_uniprot_terms()
    # 2. Instantiate a grounder with these terms
    terms_dict = defaultdict(list)
    for term in terms:
        terms_dict[term.norm_text].append(term)
    grounder = Grounder(terms_dict)
    # 3. Parse all the Protein Ontology synonyms and ground each of them, then
    # store the results
    matches_per_id = ground_proonto_terms(grounder)
    # 4. Dump spreadsheet with non-ambiguous equivalences in BioMappings format
    dump_predictions()