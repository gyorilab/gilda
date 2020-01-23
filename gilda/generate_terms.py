"""This is a script that can be run to generate a new grounding_terms.tsv file.
It uses several resource files and database clients from INDRA and requires it
to be available locally."""

import re
import os
import csv
import logging
import requests
import itertools
import indra
from indra.util import write_unicode_csv
from indra.databases import hgnc_client, uniprot_client, chebi_client, \
    go_client, mesh_client
from .term import Term
from .process import normalize
from .resources import resource_dir


indra_module_path = indra.__path__[0]
indra_resources = os.path.join(indra_module_path, 'resources')
gilda_resources = os.path.join(os.path.dirname(__file__), 'resources')

logger = logging.getLogger('gilda.generate_terms')


def read_csv(fname, header=False, delimiter='\t'):
    with open(fname, 'r') as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        if header:
            header_names = next(reader)
            for row in reader:
                yield {h: r for h, r in zip(header_names, row)}
        else:
            for row in reader:
                yield row


def generate_hgnc_terms():
    fname = os.path.join(indra_resources, 'hgnc_entries.tsv')
    logger.info('Loading %s' % fname)
    all_term_args = {}
    rows = [r for r in read_csv(fname, header=True, delimiter='\t')]
    id_name_map = {r['HGNC ID'].split(':')[1]: r['Approved symbol']
                   for r in rows}
    for row in rows:
        db, id = row['HGNC ID'].split(':')
        name = row['Approved symbol']
        # Special handling for rows representing withdrawn symbols
        if row['Status'] == 'Symbol Withdrawn':
            m = re.match(r'symbol withdrawn, see \[HGNC:(?: ?)(\d+)\]',
                         row['Approved name'])
            new_id = m.groups()[0]
            new_name = id_name_map[new_id]
            term_args = (normalize(name), name, db, new_id,
                         new_name, 'previous', 'hgnc')
            all_term_args[term_args] = None
            # NOTE: consider adding withdrawn synonyms e.g.,
            # symbol withdrawn, see pex1     symbol withdrawn, see PEX1
            # HGNC    13197   ZWS1~withdrawn  synonym
            continue
        # Handle regular entry official names
        else:
            term_args = (normalize(name), name, db, id, name, 'name', 'hgnc')
            all_term_args[term_args] = None
            if row['Approved name']:
                app_name = row['Approved name']
                term_args = (normalize(app_name), app_name, db, id, name,
                             'name', 'hgnc')
                all_term_args[term_args] = None

        # Handle regular entry synonyms
        synonyms = []
        if row['Synonyms'] and row['Synonyms']:
            synonyms += row['Synonyms'].split(', ')
        for synonym in synonyms:
            term_args = (normalize(synonym), synonym, db, id, name, 'synonym',
                         'hgnc')
            all_term_args[term_args] = None

        # Handle regular entry previous symbols
        if row['Previous symbols']:
            prev_symbols = row['Previous symbols'].split(', ')
            for prev_symbol in prev_symbols:
                term_args = (normalize(prev_symbol), prev_symbol, db, id, name,
                             'previous', 'hgnc')
                all_term_args[term_args] = None

    terms = [Term(*args) for args in all_term_args.keys()]
    logger.info('Loaded %d terms' % len(terms))
    return terms


def generate_chebi_terms():
    fname = os.path.join(indra_resources, 'chebi_entries.tsv')
    logger.info('Loading %s' % fname)
    terms = []
    for row in read_csv(fname, header=True, delimiter='\t'):
        db = 'CHEBI'
        id = 'CHEBI:' + row['CHEBI_ID']
        name = row['NAME']
        term = Term(normalize(name), name, db, id, name, 'name', 'chebi')
        terms.append(term)
    logger.info('Loaded %d terms' % len(terms))

    # Now we add synonyms
    # NOTE: this file is not in version control. The file is available
    # at ftp://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_
    # tab_delimited/names_3star.tsv.gz, it needs to be decompressed
    # into the INDRA resources folder.
    fname = os.path.join(indra_resources, 'names_3star.tsv')
    added = set()
    for row in read_csv(fname, header=True, delimiter='\t'):
        chebi_id = chebi_client.get_primary_id(str(row['COMPOUND_ID']))
        if not chebi_id:
            logger.info('Could not get valid CHEBI ID for %s' %
                        row['COMPOUND_ID'])
            continue
        db = 'CHEBI'
        id = 'CHEBI:%s' % chebi_id
        name = str(row['NAME'])
        chebi_name = \
            chebi_client.get_chebi_name_from_id(chebi_id, offline=True)
        if chebi_name is None:
            logger.info('Could not get valid name for %s' % chebi_id)
            continue

        term_args = (normalize(name), name, db, id, chebi_name, 'synonym',
                     'chebi')
        if term_args in added:
            continue
        else:
            term = Term(*term_args)
            terms.append(term)
            added.add(term_args)
    logger.info('Loaded %d terms' % len(terms))
    return terms


def generate_mesh_terms(ignore_mappings=False):
    # Load MeSH ID/label mappings
    mesh_mappings_file = os.path.join(gilda_resources, 'mesh_mappings.tsv')
    mesh_mappings = {}
    for row in read_csv(mesh_mappings_file, delimiter='\t'):
        mesh_mappings[row[1]] = (row[3], row[4])
    # Load MeSH HGNC/FPLX mappings
    mesh_names_file = os.path.join(indra_resources,
                                   'mesh_id_label_mappings.tsv')
    terms = []
    for row in read_csv(mesh_names_file, header=False, delimiter='\t'):
        db_id = row[0]
        text_name = row[1]
        mapping = mesh_mappings.get(db_id)
        if not ignore_mappings and mapping:
            db, db_id = mapping
            status = 'synonym'
            if db == 'HGNC':
                name = hgnc_client.get_hgnc_name(db_id)
            elif db == 'FPLX':
                name = db_id
        else:
            db = 'MESH'
            status = 'name'
            name = text_name
        term = Term(normalize(text_name), text_name, db, db_id, name,
                    status, 'mesh')
        terms.append(term)
        synonyms = row[2]
        if row[2]:
            synonyms = synonyms.split('|')
            for synonym in synonyms:
                term = Term(normalize(synonym), synonym, db, db_id, name,
                            'synonym', 'mesh')
                terms.append(term)
    logger.info('Loaded %d terms' % len(terms))
    return terms


def generate_go_terms():
    # TODO: add synonyms for GO terms here
    fname = os.path.join(indra_resources, 'go_id_label_mappings.tsv')
    logger.info('Loading %s' % fname)
    terms = []
    for row in read_csv(fname, delimiter='\t'):
        if not row[0].startswith('GO'):
            continue
        if 'obsolete' in row[1]:
            continue
        term = Term(normalize(row[1]), row[1], 'GO', row[0], row[1],
                    'name', 'go')
        terms.append(term)
    logger.info('Loaded %d terms' % len(terms))
    return terms


def generate_famplex_terms():
    fname = os.path.join(indra_resources, 'famplex', 'grounding_map.csv')
    logger.info('Loading %s' % fname)
    terms = []
    for row in read_csv(fname, delimiter=','):
        txt = row[0]
        norm_txt = normalize(txt)
        groundings = {k: v for k, v in zip(row[1::2], row[2::2]) if (k and v)}
        if 'FPLX' in groundings:
            id = groundings['FPLX']
            term = Term(norm_txt, txt, 'FPLX', id, id, 'assertion', 'famplex')
        elif 'HGNC' in groundings:
            id = groundings['HGNC']
            term = Term(norm_txt, txt, 'HGNC', hgnc_client.get_hgnc_id(id), id,
                        'assertion', 'famplex')
        elif 'UP' in groundings:
            db = 'UP'
            id = groundings['UP']
            name = id
            if uniprot_client.is_human(id):
                hgnc_id = uniprot_client.get_hgnc_id(id)
                if hgnc_id:
                    name = hgnc_client.get_hgnc_name(hgnc_id)
                    if hgnc_id:
                        db = 'HGNC'
                        id = hgnc_id
                else:
                    logger.warning('No gene name for %s' % id)
            term = Term(norm_txt, txt, db, id, name, 'assertion', 'famplex')
        elif 'CHEBI' in groundings:
            id = groundings['CHEBI']
            name = chebi_client.get_chebi_name_from_id(id[6:])
            term = Term(norm_txt, txt, 'CHEBI', id, name, 'assertion',
                        'famplex')
        elif 'GO' in groundings:
            id = groundings['GO']
            term = Term(norm_txt, txt, 'GO', id,
                        go_client.get_go_label(id), 'assertion', 'famplex')
        elif 'MESH' in groundings:
            id = groundings['MESH']
            term = Term(norm_txt, txt, 'MESH', id,
                        mesh_client.get_mesh_name(id), 'assertion', 'famplex')
        else:
            # TODO: handle HMDB, PUBCHEM, CHEMBL
            continue
        terms.append(term)
    return terms


def generate_uniprot_terms(download=True):
    if download:
        url = ('https://www.uniprot.org/uniprot/?format=tab&columns=id,'
               'genes(PREFERRED),protein%20names&sort=score&'
               'fil=organism:"Homo%20sapiens%20(Human)%20[9606]"'
               '%20AND%20reviewed:yes')
        logger.info('Downloading UniProt resource file')
        res = requests.get(url)
        with open('up_synonyms.tsv', 'w') as fh:
            fh.write(res.text)
    terms = []
    for row in read_csv('up_synonyms.tsv', delimiter='\t', header=True):
        names = parse_uniprot_synonyms(row['Protein names'])
        up_id = row['Entry']
        standard_name = row['Gene names  (primary )']
        ns = 'UP'
        id = row['Entry']
        # We skip a small number of not critical entries that don't have
        # standard names
        if not standard_name:
            continue
        hgnc_id = uniprot_client.get_hgnc_id(up_id)
        if hgnc_id:
            ns = 'HGNC'
            id = hgnc_id
            standard_name = hgnc_client.get_hgnc_name(hgnc_id)
        for name in names:
            # Skip names that are EC codes
            if name.startswith('EC '):
                continue
            term = Term(normalize(name), name, ns, id,
                        standard_name, 'synonym', 'uniprot')
            terms.append(term)
    return terms


def parse_uniprot_synonyms(synonyms_str):
    synonyms_str = re.sub(r'\[Includes: ([^]])+\]', '', synonyms_str).strip()
    syns = ['']
    parentheses_depth = 0
    start = True
    for idx, c in enumerate(synonyms_str):
        if c == '(':
            if start and synonyms_str[idx-1] == ' ':
                syns[-1] = syns[-1][:-1]
                syns.append('')
                start = False
            elif parentheses_depth == 0 and synonyms_str[idx-1] == ' ':
                syns[-1] = syns[-1][:-1]
                syns.append('')
            else:
                syns[-1] += c
            parentheses_depth += 1
        elif c == ')':
            if start or not start and parentheses_depth > 1:
                syns[-1] += c
            parentheses_depth -= 1
        else:
            syns[-1] += c
    return syns


def generate_adeft_terms():
    from adeft import available_shortforms
    from adeft.disambiguate import load_disambiguator
    all_term_args = set()
    for shortform in available_shortforms:
        da = load_disambiguator(shortform)
        for grounding in da.names.keys():
            if grounding == 'ungrounded' or ':' not in grounding:
                continue
            db_ns, db_id = grounding.split(':', maxsplit=1)
            if db_ns == 'HGNC':
                standard_name = hgnc_client.get_hgnc_name(db_id)
            elif db_ns == 'GO':
                standard_name = go_client.get_go_label(db_id)
            elif db_ns == 'MESH':
                standard_name = mesh_client.get_mesh_name(db_id)
            elif db_ns == 'CHEBI':
                standard_name = chebi_client.get_chebi_name_from_id(db_id)
            elif db_ns == 'FPLX':
                standard_name = db_id
            elif db_ns == 'UP':
                standard_name = uniprot_client.get_gene_name(db_id)
            else:
                logger.warning('Unknown grounding namespace from Adeft: %s' %
                               db_ns)
                continue
            term_args = (normalize(shortform), shortform, db_ns, db_id,
                         standard_name, 'synonym', 'adeft')
            all_term_args.add(term_args)
    terms = [Term(*term_args) for term_args in sorted(list(all_term_args),
                                                      key=lambda x: x[0])]
    return terms


def filter_out_duplicates(terms):
    logger.info('Filtering %d terms for uniqueness...' % len(terms))
    term_key = lambda term: (term.db, term.id, term.text)
    statuses = {'assertion': 1, 'name': 2, 'synonym': 3, 'previous': 4}
    new_terms = []
    for _, terms in itertools.groupby(sorted(terms, key=lambda x: term_key(x)),
                                      key=lambda x: term_key(x)):
        terms = sorted(terms, key=lambda x: statuses[x.status])
        new_terms.append(terms[0])
    # Re-sort the terms
    new_terms = sorted(new_terms, key=lambda x: (x.text, x.db, x.id))
    logger.info('Got %d unique terms...' % len(new_terms))
    return new_terms


def get_all_terms():
    terms = generate_famplex_terms()
    terms += generate_hgnc_terms()
    terms += generate_chebi_terms()
    terms += generate_go_terms()
    terms += generate_mesh_terms()
    terms += generate_uniprot_terms()
    terms += generate_adeft_terms()
    terms = filter_out_duplicates(terms)
    return terms


if __name__ == '__main__':
    terms = get_all_terms()
    fname = os.path.join(resource_dir, 'grounding_terms.tsv')
    logger.info('Dumping into %s' % fname)
    write_unicode_csv(fname, [t.to_list() for t in terms], delimiter='\t')
