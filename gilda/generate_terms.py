"""This is a script that can be run to generate a new grounding_terms.tsv file.
It uses several resource files and database clients from INDRA and requires it
to be available locally."""

import re
import os
import pandas
import logging
import indra
from indra.util import write_unicode_csv
from indra.databases import hgnc_client, uniprot_client, chebi_client, \
    go_client, mesh_client
from .term import Term
from .process import normalize
from .resources import resource_dir


indra_module_path = indra.__path__[0]
resources = os.path.join(indra_module_path, 'resources')

logger = logging.getLogger('gilda.generate_terms')


def generate_hgnc_terms():
    fname = os.path.join(resources, 'hgnc_entries.tsv')
    logger.info('Loading %s' % fname)
    df = pandas.read_csv(fname, delimiter='\t', dtype='str')
    all_term_args = dict()
    for idx, row in df.iterrows():
        db, id = row['HGNC ID'].split(':')
        name = row['Approved symbol']
        # Special handling for rows representing withdrawn symbols
        if 'withdrawn' in name:
            match = re.match(r'([^ ]+)~withdrawn', name)
            if not match:
                match = re.match(r'([^ ]+)~withdrawn,synonym', name)
                if not match:
                    continue
            previous_name = match.groups()[0]
            match = re.match(r'symbol withdrawn, see ([^ ]+)',
                             row['Approved name'])
            if not match:
                continue
            new_name = match.groups()[0]
            new_id = hgnc_client.get_hgnc_id(new_name)
            if not new_id:
                continue
            term_args = (normalize(previous_name), previous_name, db, new_id,
                         new_name, 'previous')
            all_term_args[term_args] = None
            # NOTE: consider adding withdrawn synonyms e.g.,
            # symbol withdrawn, see pex1     symbol withdrawn, see PEX1
            # HGNC    13197   ZWS1~withdrawn  synonym
            continue
        # Handle regular entry official names
        else:
            term_args = (normalize(name), name, db, id, name, 'name')
            all_term_args[term_args] = None

        # Handle regular entry synonyms
        synonyms = []
        if row['Approved name']:
            synonyms.append(row['Approved name'])
        if row['Synonyms'] and not pandas.isnull(row['Synonyms']):
            synonyms += row['Synonyms'].split(', ')
        for synonym in synonyms:
            term_args = (normalize(synonym), synonym, db, id, name, 'synonym')
            all_term_args[term_args] = None

        # Handle regular entry previous symbols
        if not pandas.isna(row['Previous symbols']):
            prev_symbols = row['Previous symbols'].split(', ')
            for prev_symbol in prev_symbols:
                term_args = (normalize(prev_symbol), prev_symbol, db, id, name,
                             'previous')
                all_term_args[term_args] = None

    terms = [Term(*args) for args in all_term_args.keys()]
    logger.info('Loaded %d terms' % len(terms))
    return terms


def generate_chebi_terms():
    fname = os.path.join(resources, 'chebi_entries.tsv')
    logger.info('Loading %s' % fname)
    df = pandas.read_csv(fname, delimiter='\t', dtype='str')
    terms = []
    for idx, row in df.iterrows():
        db = 'CHEBI'
        id = 'CHEBI:' + row['CHEBI_ID']
        name = row['NAME']
        term = Term(normalize(name), name, db, id, name, 'name')
        terms.append(term)
    logger.info('Loaded %d terms' % len(terms))

    # Now we add synonyms
    # NOTE: this file is not in version control. The file is available
    # at ftp://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_
    # tab_delimited/names_3star.tsv.gz, it needs to be decompressed
    # into the INDRA resources folder.
    fname = os.path.join(resources, 'names_3star.tsv')
    df = pandas.read_csv(fname, delimiter='\t', dtype='str',
                         keep_default_na=False, na_values=[''])
    added = set()
    for idx, row in df.iterrows():
        chebi_id = chebi_client.get_primary_id(str(row['COMPOUND_ID']))
        db = 'CHEBI'
        id = 'CHEBI:%s' % chebi_id
        name = str(row['NAME'])
        chebi_name = \
            chebi_client.get_chebi_name_from_id(chebi_id, offline=True)
        term_args = (normalize(name), name, db, id, chebi_name, 'synonym')
        if term_args in added:
            continue
        else:
            term = Term(*term_args)
            terms.append(term)
            added.add(term_args)
    logger.info('Loaded %d terms' % len(terms))

    return terms


def generate_mesh_terms():
    fname = os.path.join(resources, 'mesh_id_label_mappings.tsv')
    logger.info('Loading %s' % fname)
    df = pandas.read_csv(fname, delimiter='\t', dtype='str', header=None,
                         keep_default_na=False, na_values=None)
    terms = []
    for idx, row in df.iterrows():
        db = 'MESH'
        id_ = row[0]
        name = row[1]
        term = Term(normalize(name), name, db, id_, name, 'name')
        terms.append(term)
        synonyms = row[2]
        if row[2]:
            synonyms = synonyms.split('|')
            for synonym in synonyms:
                term = Term(normalize(synonym), synonym, db, id_, name,
                            'synonym')
                terms.append(term)
    logger.info('Loaded %d terms' % len(terms))
    return terms


def generate_go_terms():
    # TODO: add synonyms for GO terms here
    fname = os.path.join(resources, 'go_id_label_mappings.tsv')
    logger.info('Loading %s' % fname)
    df = pandas.read_csv(fname, delimiter='\t', dtype='str', header=None)
    terms = []
    for idx, row in df.iterrows():
        if not row[0].startswith('GO'):
            continue
        if 'obsolete' in row[1]:
            continue
        term = Term(normalize(row[1]), row[1], 'GO', row[0], row[1], 'name')
        terms.append(term)
    logger.info('Loaded %d terms' % len(terms))
    return terms


def generate_famplex_terms():
    fname = os.path.join(resources, 'famplex', 'grounding_map.csv')
    logger.info('Loading %s' % fname)
    df = pandas.read_csv(fname, delimiter=',', dtype='str', header=None)
    terms = []
    for idx, row in df.iterrows():
        txt = row[0]
        norm_txt = normalize(txt)
        groundings = {k: v for k, v in zip(row[1::2], row[2::2]) if
                      (not pandas.isnull(k) and not pandas.isnull(v))}
        if 'FPLX' in groundings:
            id = groundings['FPLX']
            term = Term(norm_txt, txt, 'FPLX', id, id, 'assertion')
        elif 'HGNC' in groundings:
            id = groundings['HGNC']
            term = Term(norm_txt, txt, 'HGNC', hgnc_client.get_hgnc_id(id), id,
                        'assertion')
        elif 'UP' in groundings:
            db = 'UP'
            id = groundings['UP']
            name = id
            gene_name = uniprot_client.get_gene_name(id)
            if gene_name:
                name = gene_name
                hgnc_id = hgnc_client.get_hgnc_id(gene_name)
                if hgnc_id:
                    db = 'HGNC'
                    id = hgnc_id
            term = Term(norm_txt, txt, db, id, name, 'assertion')
        elif 'CHEBI' in groundings:
            id = groundings['CHEBI']
            name = chebi_client.get_chebi_name_from_id(id[6:])
            term = Term(norm_txt, txt, 'CHEBI', id, name, 'assertion')
        elif 'GO' in groundings:
            id = groundings['GO']
            term = Term(norm_txt, txt, 'GO', id,
                        go_client.get_go_label(id), 'assertion')
        elif 'MESH' in groundings:
            id = groundings['MESH']
            term = Term(norm_txt, txt, 'MESH', id,
                        mesh_client.get_mesh_name(id), 'assertion')
        else:
            # TODO: handle HMDB, PUBCHEM, CHEMBL
            continue
        terms.append(term)
    return terms


def get_all_terms():
    terms = generate_famplex_terms()
    terms += generate_hgnc_terms()
    terms += generate_chebi_terms()
    terms += generate_go_terms()
    terms += generate_mesh_terms()
    return terms


if __name__ == '__main__':
    terms = get_all_terms()
    fname = os.path.join(resource_dir, 'grounding_terms.tsv')
    logger.info('Dumping into %s' % fname)
    write_unicode_csv(fname, [t.to_list() for t in terms], delimiter='\t')
