"""This is a script that can be run to generate a new grounding_terms.tsv file.
It uses several resource files and database clients from INDRA and requires it
to be available locally."""

import re
import os
import csv
import json
import logging
import requests
import itertools
import indra
from indra.util import write_unicode_csv
from indra.databases import hgnc_client, uniprot_client, chebi_client, \
    go_client, mesh_client, doid_client
from .term import Term
from .process import normalize
from .resources import resource_dir


indra_module_path = indra.__path__[0]
indra_resources = os.path.join(indra_module_path, 'resources')

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
        if row['Alias symbols']:
            synonyms += row['Alias symbols'].split(', ')
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
    if not os.path.exists(fname):
        import pandas as pd
        chebi_url = 'ftp://ftp.ebi.ac.uk/pub/databases/chebi/' \
                    'Flat_file_tab_delimited/names_3star.tsv.gz'
        logger.info('Loading %s into memory. You can download and decompress'
                    ' it in the indra/resources folder for faster access.'
                    % chebi_url)
        df = pd.read_csv(chebi_url, sep='\t')
        rows = (row for _, row in df.iterrows())
    else:
        rows = read_csv(fname, header=True, delimiter='\t')

    added = set()
    for row in rows:
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
    # Load MeSH HGNC/FPLX mappings
    mesh_names_file = os.path.join(indra_resources,
                                   'mesh_id_label_mappings.tsv')
    terms = []
    for row in read_csv(mesh_names_file, header=False, delimiter='\t'):
        db_id = row[0]
        text_name = row[1]
        mapping = mesh_mappings.get(db_id)
        if not ignore_mappings and mapping and mapping[0] \
                not in {'EFO', 'HP', 'DOID'}:
            db, db_id, name = mapping
            status = 'synonym'
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
            mesh_mapping = mesh_mappings.get(id)
            db, db_id, name = mesh_mapping if mesh_mapping else \
                ('MESH', id, mesh_client.get_mesh_name(id))
            term = Term(norm_txt, txt, db, db_id, name, 'assertion', 'famplex')
        else:
            # TODO: handle HMDB, PUBCHEM, CHEMBL
            continue
        terms.append(term)
    return terms


def generate_uniprot_terms(download=False):
    path = os.path.join(resource_dir, 'up_synonyms.tsv')
    if not os.path.exists(path) or download:
        url = ('https://www.uniprot.org/uniprot/?format=tab&columns=id,'
               'genes(PREFERRED),protein%20names&sort=score&'
               'fil=organism:"Homo%20sapiens%20(Human)%20[9606]"'
               '%20AND%20reviewed:yes')
        logger.info('Downloading UniProt resource file')
        res = requests.get(url)
        with open(path, 'w') as fh:
            fh.write(res.text)
    terms = []
    for row in read_csv(path, delimiter='\t', header=True):
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
    synonyms_str = re.sub(r'\[Includes: ([^]])+\]',
                          '', synonyms_str).strip()
    synonyms_str = re.sub(r'\[Cleaved into: ([^]])+\]',
                          '', synonyms_str).strip()

    def find_block_from_right(s):
        parentheses_depth = 0
        assert s.endswith(')')
        s = s[:-1]
        block = ''
        for c in s[::-1]:
            if c == ')':
                parentheses_depth += 1
            elif c == '(':
                if parentheses_depth > 0:
                    parentheses_depth -= 1
                else:
                    return block
            block = c + block
        return block

    syns = []
    while True:
        if not synonyms_str:
            return syns
        if not synonyms_str.endswith(')'):
            return [synonyms_str] + syns

        syn = find_block_from_right(synonyms_str)
        syns = [syn] + syns
        synonyms_str = synonyms_str[:-len(syn)-3]


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


def generate_doid_terms():
    return _generate_obo_terms('doid')


def generate_efo_terms():
    return _generate_obo_terms('efo')


def generate_hp_terms():
    return _generate_obo_terms('hp')


def _generate_obo_terms(prefix):
    filename = os.path.join(indra_resources, '%s.json' % prefix)
    logger.info('Loading %s', filename)
    with open(filename) as file:
        entries = json.load(file)

    terms = []
    for entry in entries:
        db, db_id, name = prefix.upper(), entry['id'], entry['name']
        # We first need to decide if we prioritize another name space
        xref_dict = {xr['namespace']: xr['id'] for xr in entry['xrefs']}
        # Handle MeSH mappings first
        auto_mesh_mapping = mesh_mappings_reverse.get((db, db_id))
        if auto_mesh_mapping:
            db, db_id, name = ('MESH', auto_mesh_mapping[0],
                               auto_mesh_mapping[1])
        elif 'MESH' in xref_dict or 'MSH' in xref_dict:
            mesh_id = xref_dict.get('MESH') or xref_dict.get('MSH')
            # Since we currently only include regular MeSH terms (which start
            # with D), we only need to do the mapping if that's the case.
            # We don't map any supplementary terms that start with C.
            if mesh_id.startswith('D'):
                mesh_name = mesh_client.get_mesh_name(mesh_id)
                if mesh_name:
                    # Here we need to check if we further map the MeSH ID to
                    # another namespace
                    mesh_mapping = mesh_mappings.get(mesh_id)
                    db, db_id, name = mesh_mapping if (mesh_mapping and \
                            mesh_mapping[0] not in {'EFO', 'HP', 'DOID'}) else \
                        ('MESH', mesh_id, mesh_name)
        # Next we look at mappings to DOID
        # TODO: are we sure that the DOIDs that we get here (from e.g., EFO)
        # cannot be mapped further to MeSH per the DOID resource file?
        elif 'DOID' in xref_dict:
            doid = xref_dict['DOID']
            if not doid.startswith('DOID:'):
                doid = 'DOID:' + doid
            doid_prim_id = doid_client.get_doid_id_from_doid_alt_id(doid)
            if doid_prim_id:
                doid = doid_prim_id
            doid_name = doid_client.get_doid_name_from_doid_id(doid)
            db, db_id, name = 'DOID', doid, doid_name

        # Add a term for the name first
        name_term = Term(
            norm_text=normalize(name),
            text=name,
            db=db,
            id=db_id,
            entry_name=name,
            status='name',
            source=prefix,
        )
        terms.append(name_term)

        # Then add all the synonyms
        for synonym in set(entry['synonyms']):
            # Some synonyms are tagged as ambiguous, we remove these
            if 'ambiguous' in synonym.lower():
                continue
            # Some synonyms contain a "formerly" clause, we remove these
            match = re.match(r'(.+) \(formerly', synonym)
            if match:
                synonym = match.groups()[0]
            # Some synonyms contain additional annotations
            # e.g. Hyperplasia of facial adipose tissue" NARROW
            # [ORCID:0000-0001-5889-4463]
            # If this is the case, we strip these off
            match = re.match(r'([^"]+)', synonym)
            if match:
                synonym = match.groups()[0]

            synonym_term = Term(
                norm_text=normalize(synonym),
                text=synonym,
                db=db,
                id=db_id,
                entry_name=name,
                status='synonym',
                source=prefix,
            )
            terms.append(synonym_term)

    logger.info('Loaded %d terms from %s', len(terms), prefix)
    return terms


def _make_mesh_mappings():
    # Load MeSH ID/label mappings
    from .resources import MESH_MAPPINGS_PATH
    mesh_mappings = {}
    mesh_mappings_reverse = {}
    for row in read_csv(MESH_MAPPINGS_PATH, delimiter='\t'):
        # We can skip row[2] which is the MeSH standard name for the entry
        mesh_mappings[row[1]] = row[3:]
        mesh_mappings_reverse[(row[3], row[4])] = [row[1], row[2]]
    return mesh_mappings, mesh_mappings_reverse


mesh_mappings, mesh_mappings_reverse = _make_mesh_mappings()


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
    terms = []

    generated_term_groups = [
        generate_famplex_terms(),
        generate_hgnc_terms(),
        generate_chebi_terms(),
        generate_go_terms(),
        generate_mesh_terms(),
        generate_uniprot_terms(),
        generate_adeft_terms(),
        generate_doid_terms(),
        generate_hp_terms(),
        generate_efo_terms(),
    ]
    for generated_terms in generated_term_groups:
        terms += generated_terms

    terms = filter_out_duplicates(terms)
    return terms


def main():
    terms = get_all_terms()
    from .resources import GROUNDING_TERMS_PATH as fname
    logger.info('Dumping into %s' % fname)
    write_unicode_csv(fname, [t.to_list() for t in terms], delimiter='\t')


if __name__ == '__main__':
    main()
