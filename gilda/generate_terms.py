"""This is a script that can be run to generate a new grounding_terms.tsv file.
It uses several resource files and database clients from INDRA and requires it
to be available locally."""

import re
import os
import csv
import json
import logging
import requests
import indra
from indra.databases import hgnc_client, uniprot_client, chebi_client, \
    go_client, mesh_client, doid_client
from indra.statements.resources import amino_acids
from .term import Term, dump_terms, filter_out_duplicates
from .process import normalize
from .resources import resource_dir, popular_organisms


indra_module_path = indra.__path__[0]
indra_resources = os.path.join(indra_module_path, 'resources')

logger = logging.getLogger('gilda.generate_terms')


def read_csv(fname, header=False, delimiter='\t', quotechar='"'):
    with open(fname, 'r') as fh:
        reader = csv.reader(fh, delimiter=delimiter, quotechar=quotechar)
        if header:
            header_names = next(reader)
            for row in reader:
                yield {h: r for h, r in zip(header_names, row)}
        else:
            for row in reader:
                yield row


def generate_hgnc_terms():
    fname = os.path.join(resource_dir, 'hgnc_entries.tsv')

    if not os.path.exists(fname):
        # Select relevant columns and parameters
        cols = [
            'gd_hgnc_id', 'gd_app_sym', 'gd_app_name', 'gd_status',
            'gd_aliases', 'gd_prev_sym', 'gd_name_aliases'
        ]

        statuses = ['Approved', 'Entry%20Withdrawn']
        params = {
                'hgnc_dbtag': 'on',
                'order_by': 'gd_app_sym_sort',
                'format': 'text',
                'submit': 'submit'
                }

        # Construct a download URL from the above parameters
        url = 'https://www.genenames.org/cgi-bin/download/custom?'
        url += '&'.join(['col=%s' % c for c in cols]) + '&'
        url += '&'.join(['status=%s' % s for s in statuses]) + '&'
        url += '&'.join(['%s=%s' % (k, v) for k, v in params.items()])

        # Download the file
        logger.info('Downloading HGNC resource file')
        res = requests.get(url)
        with open(fname, 'w') as fh:
            fh.write(res.text)

    logger.info('Loading %s' % fname)
    all_term_args = {}
    rows = [r for r in read_csv(fname, header=True, delimiter='\t',
                                quotechar=None)]
    id_name_map = {r['HGNC ID'].split(':')[1]: r['Approved symbol']
                   for r in rows}
    organism = '9606'  # human
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
                         new_name, 'former_name', 'hgnc', organism)
            all_term_args[term_args] = None
            # NOTE: consider adding withdrawn synonyms e.g.,
            # symbol withdrawn, see pex1     symbol withdrawn, see PEX1
            # HGNC    13197   ZWS1~withdrawn  synonym
            continue
        # Handle regular entry official names
        else:
            term_args = (normalize(name), name, db, id, name, 'name', 'hgnc',
                         organism)
            all_term_args[term_args] = None
            if row['Approved name']:
                app_name = row['Approved name']
                term_args = (normalize(app_name), app_name, db, id, name,
                             'name', 'hgnc', organism)
                all_term_args[term_args] = None

        # Handle regular entry synonyms
        synonyms = []
        if row['Alias symbols']:
            synonyms += row['Alias symbols'].split(', ')
        for synonym in synonyms:
            term_args = (normalize(synonym), synonym, db, id, name, 'synonym',
                         'hgnc', organism)
            all_term_args[term_args] = None

        # Handle regular entry previous symbols
        if row['Previous symbols']:
            prev_symbols = row['Previous symbols'].split(', ')
            for prev_symbol in prev_symbols:
                term_args = (normalize(prev_symbol), prev_symbol, db, id, name,
                             'former_name', 'hgnc', organism)
                all_term_args[term_args] = None

        if row['Alias names']:
            for alias_name in extract_hgnc_alias_names(row['Alias names']):
                alias_name = alias_name.strip()
                # There are double quotes and sometimes spurious extra spaces
                term_args = (normalize(alias_name), alias_name, db, id, name,
                             'synonym', 'synonym', organism)
                all_term_args[term_args] = None

    terms = [Term(*args) for args in all_term_args.keys()]
    logger.info('Loaded %d terms' % len(terms))
    return terms


def extract_hgnc_alias_names(alias_str):
    # The string is a comma-separated list of aliases each within
    # double quotes, and commas can appear within double quotes as well.
    if re.match(r'^"([^"]+)"$', alias_str):
        names = [alias_str.strip('"').strip()]
    else:
        names = [s.strip() for s in
                 next(csv.reader([alias_str], skipinitialspace=True))]
    return names


def generate_chebi_terms():
    # We can get standard names directly from the OBO
    terms = _generate_obo_terms('chebi', ignore_mappings=True,
                                map_to_ns={})

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
        name = str(row['NAME'])
        chebi_name = \
            chebi_client.get_chebi_name_from_id(chebi_id, offline=True)
        if chebi_name is None:
            logger.info('Could not get valid name for %s' % chebi_id)
            continue
        # We skip entries of the form Glu-Lys with synonyms like EK since
        # there are highly ambiguous with other acronyms, and are unlikely
        # to be used in practice.
        if is_aa_sequence(chebi_name) and re.match(r'(^[A-Z-]+$)', name):
            continue

        term_args = (normalize(name), name, db, chebi_id, chebi_name, 'synonym',
                     'chebi')
        if term_args in added:
            continue
        else:
            term = Term(*term_args)
            terms.append(term)
            added.add(term_args)
    logger.info('Loaded %d terms' % len(terms))
    return terms


def is_aa_sequence(txt):
    """Return True if the given text is a sequence of amino acids like Tyr-Glu.
    """
    return ('-' in txt) and (all(part in aa_abbrevs
                                 for part in txt.split('-')))


aa_abbrevs = {aa['short_name'].capitalize() for aa in amino_acids.values()}


def generate_mesh_terms(ignore_mappings=False):
    mesh_name_files = ['mesh_id_label_mappings.tsv',
                       'mesh_supp_id_label_mappings.tsv']
    terms = []
    for fname in mesh_name_files:
        logger.info('Loading %s' % fname)
        mesh_names_file = os.path.join(indra_resources, fname)
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
                        status, 'mesh',
                        source_db='MESH' if db != 'MESH' else None,
                        source_id=row[0] if db != 'MESH' else None)
            terms.append(term)
            synonyms = row[2]
            if row[2]:
                synonyms = synonyms.split('|')
                for synonym in synonyms:
                    term = Term(normalize(synonym), synonym, db, db_id, name,
                                'synonym', 'mesh',
                                source_db='MESH' if db != 'MESH' else None,
                                source_id=row[0] if db != 'MESH' else None)
                    terms.append(term)
        logger.info('Loaded %d terms' % len(terms))
    return terms


def generate_go_terms():
    fname = os.path.join(indra_resources, 'go.json')
    logger.info('Loading %s' % fname)
    with open(fname, 'r') as fh:
        entries = json.load(fh)
    terms = []
    for entry in entries:
        go_id = entry['id']
        name = entry['name']
        # First handle the name term
        term = Term(normalize(name), name, 'GO', go_id, name, 'name', 'go')
        terms.append(term)
        # Next look at synonyms, sometimes those match the name so we
        # deduplicate
        for synonym in set(entry.get('synonyms', [])) - {name}:
            # GO includes around 40k synonyms for terms that represent
            # activity out of which around 5k are actually
            # synonyms for entities. One example is "EGFR" as a synonym
            # for "epidermal growth factor-activated receptor activity".
            # We skip these according to the following logic.
            if 'activity' in name and 'activity' not in synonym:
                continue
            term = Term(normalize(synonym), synonym, 'GO', go_id, name,
                        'synonym', 'go')
            terms.append(term)
    logger.info('Loaded %d terms' % len(terms))
    return terms


def generate_famplex_terms(ignore_mappings=False):
    fname = os.path.join(indra_resources, 'famplex', 'grounding_map.csv')
    logger.info('Loading %s' % fname)
    terms = []
    for row in read_csv(fname, delimiter=','):
        txt = row[0]
        norm_txt = normalize(txt)
        groundings = {k: v for k, v in zip(row[1::2], row[2::2]) if (k and v)}
        if 'FPLX' in groundings:
            id = groundings['FPLX']
            term = Term(norm_txt, txt, 'FPLX', id, id, 'curated', 'famplex')
        elif 'HGNC' in groundings:
            id = groundings['HGNC']
            term = Term(norm_txt, txt, 'HGNC', hgnc_client.get_hgnc_id(id), id,
                        'curated', 'famplex', '9606')
        elif 'UP' in groundings:
            db = 'UP'
            id = groundings['UP']
            name = id
            organism = None
            if uniprot_client.is_human(id):
                organism = '9606'
                hgnc_id = uniprot_client.get_hgnc_id(id)
                if hgnc_id:
                    name = hgnc_client.get_hgnc_name(hgnc_id)
                    if hgnc_id:
                        db = 'HGNC'
                        id = hgnc_id
                else:
                    logger.warning('No gene name for %s' % id)
            # FIXME: we should figure out what organism the given protein
            # comes from and then add that organism info, otherwise
            # these groundings will be asserted even if the organism
            # doesn't match
            term = Term(norm_txt, txt, db, id, name, 'curated', 'famplex',
                        organism)
        elif 'CHEBI' in groundings:
            id = groundings['CHEBI']
            name = chebi_client.get_chebi_name_from_id(id[6:])
            term = Term(norm_txt, txt, 'CHEBI', id, name, 'curated',
                        'famplex')
        elif 'GO' in groundings:
            id = groundings['GO']
            term = Term(norm_txt, txt, 'GO', id,
                        go_client.get_go_label(id), 'curated', 'famplex')
        elif 'MESH' in groundings:
            id = groundings['MESH']
            mesh_mapping = mesh_mappings.get(id)
            db, db_id, name = mesh_mapping if (mesh_mapping
                                               and not ignore_mappings) else \
                ('MESH', id, mesh_client.get_mesh_name(id))
            term = Term(norm_txt, txt, db, db_id, name, 'curated', 'famplex',
                        None, 'MESH', groundings['MESH'])
        else:
            # TODO: handle HMDB, PUBCHEM, CHEMBL
            continue
        terms.append(term)
    return terms


def generate_uniprot_terms(download=False, organisms=None):
    if not organisms:
        organisms = popular_organisms
    path = os.path.join(resource_dir, 'up_synonyms.tsv')
    if not os.path.exists(path) or download:
        # Columns according to the new API, comments for old API
        columns = [
            'accession',     # id
            'gene_primary',  # genes(PREFERRED)
            'gene_synonym',  # genes(ALTERNATIVE)'
            'protein_name',  # protein names
            'organism_id',   # organism-id
        ]
        org_filter_str = '+OR+'.join(f'(taxonomy_id:{org})' for org in organisms)
        query = f'reviewed:true+AND+({org_filter_str})'
        url = (f'https://rest.uniprot.org/uniprotkb/stream?'
               f'format=tsv&'
               f'query={query}&'
               f'compressed=false&'
               f'fields={",".join(columns)}')
        logger.info('Downloading UniProt resource file')
        res = requests.get(url)
        with open(path, 'w') as fh:
            fh.write(res.text)
    terms = []
    logger.info('Loading %s' % path)
    for row in read_csv(path, delimiter='\t', header=True):
        terms += get_terms_from_uniprot_row(row)

    return terms


def get_terms_from_uniprot_row(row):
    terms = []
    up_id = row['Entry']
    organism = row['Organism (ID)']

    # As of 3/2/2022 there is an error in UniProt data that we need to manually
    # patch here
    if up_id == 'Q2QKR2':
        row['Protein names'] = row['Protein names'][:-1]
    protein_names = parse_uniprot_synonyms(row['Protein names'])

    # These two lists are aligned and each separated by "; " if there
    # are multiple genes. If there are no genes listed, we simply have
    # an empty string here. If there are multiple genes but one doesn't
    # have a name given, a corresponding "; " placeholder is still there.
    # Consequently, we encounter cases like
    # P34539»·; »·; »·
    # where there are two genes but neither of them have names or
    # synonyms listed.
    primary_gene_names = row['Gene Names (primary)'].split('; ')
    gene_synonyms = row['Gene Names (synonym)'].split('; ')

    multi_gene = len(primary_gene_names) > 1

    # We generally use the gene name as the standard name
    # except when there are multiple gene names or the gene name is missing
    if not primary_gene_names or multi_gene:
        standard_name = protein_names[0]
    else:
        standard_name = primary_gene_names[0]
    # We skip a small number of non-critical entries that don't have
    # standard names
    if not standard_name:
        return []

    # By default, we use the UniProt namespace and ID
    ns = 'UP'
    id = up_id
    # For human genes, we resolve redundancies by mapping to HGNC
    hgnc_id = uniprot_client.get_hgnc_id(up_id)
    # We only map to HGNC if there is a single gene corresponding to
    # this protein. Otherwise, we keep the UniProt ID.
    if hgnc_id and not multi_gene:
        ns = 'HGNC'
        id = hgnc_id
        standard_name = hgnc_client.get_hgnc_name(hgnc_id)

    # We add all of the protein names as synonyms
    for name in protein_names:
        # Skip names that are EC codes
        if name.startswith('EC '):
            continue
        if name == standard_name:
            continue
        term = Term(normalize(name), name, ns, id,
                    standard_name, 'synonym', 'uniprot',
                    organism, None if ns == 'UP' else 'UP',
                    None if ns == 'UP' else up_id)
        terms.append(term)

    # We add the standard name (usually the gene name)
    term = Term(normalize(standard_name), standard_name,
                ns, id, standard_name, 'name', 'uniprot',
                organism, None if ns == 'UP' else 'UP',
                None if ns == 'UP' else up_id)
    terms.append(term)

    # If we have gene synonyms we include them according to the following logic.
    # For human genes we do not include synonyms if there are multiple genes
    # corresponding to this protein. For non-human genes, since we don't
    # have a separate gene resource, we do include gene names and synonyms
    # even if there are multiple genes corresponding to the protein.
    if not ((organism == '9606') and multi_gene):
        for gene_name, gene_synonyms_str in zip(primary_gene_names,
                                                gene_synonyms):
            all_synonyms = gene_synonyms_str.split(' ')
            # We can skip a gene name if we used it as standard name
            if gene_name and (gene_name != standard_name):
                all_synonyms.append(gene_name)
            for synonym in all_synonyms:
                if not synonym:
                    continue
                term = Term(normalize(synonym), synonym,
                            ns, id, standard_name, 'synonym', 'uniprot',
                            organism, None if ns == 'UP' else 'UP',
                            None if ns == 'UP' else up_id)
                terms.append(term)
    return terms


ec_code_pattern = re.compile(r'EC [\d\.-]+')


def parse_uniprot_synonyms(synonyms_str):
    synonyms_str = re.sub(r'\[Includes: ([^]])+\]',
                          '', synonyms_str).strip()
    synonyms_str = re.sub(r'\[Cleaved into: ([^]])+\]( \(Fragments\))?',
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
        # If the string is empty at this point, we return with the
        # synonyms so far
        if not synonyms_str:
            return syns
        # If the string doesn't end with a parenthesis, that means it's
        # the first synonym which isn't in parentheses, and so we prepend
        # it to the list of synonyms and return
        if not synonyms_str.endswith(')'):
            return [synonyms_str] + syns

        syn = find_block_from_right(synonyms_str)
        # This is where we remove the synonym we just processed plus
        # the space and opening/closing parens (3 characters) to get to the
        # next synonym
        to_remove = len(syn) + 3
        # One corner case is when there happens to be a parenthesis in the
        # first synonym in the list like "X(0) (...) (...)"
        # In this case the entire synonyms_str is to be used and we can
        # prepend and return
        if synonyms_str[-to_remove] != ' ':
            return [synonyms_str] + syns

        # EC codes are not valid synonyms bot otherwise we prepend
        # the synonym to the list of synonyms
        if not ec_code_pattern.match(syn):
            syns = [syn] + syns
        # We now remove the processed suffix
        synonyms_str = synonyms_str[:-to_remove]


def generate_adeft_terms():
    from adeft import available_shortforms
    from adeft.disambiguate import load_disambiguator
    from indra.ontology.standardize import get_standard_name
    all_term_args = set()
    add_prefix = ['BTO', 'HP', 'DOID']
    remove_prefix = ['EFO', 'NCIT', 'OMIT']
    for shortform in available_shortforms:
        da = load_disambiguator(shortform)
        for grounding, name in da.names.items():
            if grounding == 'ungrounded' or ':' not in grounding:
                continue
            db_ns, db_id = grounding.split(':', maxsplit=1)
            if db_ns in remove_prefix:
                if db_id.startswith(db_ns + ':'):
                    db_id = db_id[len(db_ns) + 1:]
            if db_ns in add_prefix:
                if not db_id.startswith(db_ns + ':'):
                    db_id = db_ns + ':' + db_id
            if db_id == 'PF00112)':
                db_id = 'PF00112'
            # Here we do a name standardization via INDRA just in case
            # there is a discrepancy
            indra_standard_name = get_standard_name({db_ns: db_id})
            if indra_standard_name:
                name = indra_standard_name
            term_args = (normalize(shortform), shortform, db_ns, db_id,
                         name, 'synonym', 'adeft')
            all_term_args.add(term_args)
    terms = [Term(*term_args) for term_args in sorted(list(all_term_args),
                                                      key=lambda x: x[0])]
    return terms


def generate_doid_terms(ignore_mappings=False):
    return _generate_obo_terms('doid', ignore_mappings)


def generate_efo_terms(ignore_mappings=False):
    terms = _generate_obo_terms('efo', ignore_mappings)
    # We remove BFO terms since they are too generic to be useful
    terms = [t for t in terms if not t.id.startswith('BFO:')]
    return terms


def generate_hp_terms(ignore_mappings=False):
    return _generate_obo_terms('hp', ignore_mappings)


def terms_from_obo_json_entry(entry, prefix, ignore_mappings=False,
                              map_to_ns=None):
    if map_to_ns is None:
        map_to_ns = {'MESH', 'DOID'}
    terms = []
    db, db_id, name = prefix.upper(), entry['id'], entry['name']
    # We first need to decide if we prioritize another name space
    xref_dict = {xr['namespace']: xr['id'] for xr in entry.get('xrefs', [])}
    # Handle MeSH mappings first
    auto_mesh_mapping = mesh_mappings_reverse.get((db, db_id))
    if auto_mesh_mapping and not ignore_mappings:
        db, db_id, name = ('MESH', auto_mesh_mapping[0],
                           auto_mesh_mapping[1])
    elif 'MESH' in map_to_ns and ('MESH' in xref_dict or 'MSH' in xref_dict):
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
                db, db_id, name = mesh_mapping if \
                    (mesh_mapping and (mesh_mapping[0]
                                       not in {'EFO', 'HP', 'DOID'})) \
                    else ('MESH', mesh_id, mesh_name)
    # Next we look at mappings to DOID
    # TODO: are we sure that the DOIDs that we get here (from e.g., EFO)
    # cannot be mapped further to MeSH per the DOID resource file?
    elif 'DOID' in map_to_ns and 'DOID' in xref_dict:
        doid = xref_dict['DOID']
        if not doid.startswith('DOID:'):
            doid = 'DOID:' + doid
        doid_prim_id = doid_client.get_doid_id_from_doid_alt_id(doid)
        if doid_prim_id:
            doid = doid_prim_id
        doid_name = doid_client.get_doid_name_from_doid_id(doid)
        # If we don't get a name here, it's likely because an entry is
        # obsolete so we don't do the mapping
        if doid_name:
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
        source_db=prefix.upper() if db != prefix.upper() else None,
        source_id=entry['id'] if db != prefix.upper() else None,
    )
    terms.append(name_term)

    # Then add all the synonyms
    for synonym in set(entry.get('synonyms', [])):
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
            source_db=prefix.upper() if db != prefix.upper() else None,
            source_id=entry['id'] if db != prefix.upper() else None,
        )
        terms.append(synonym_term)
    return terms


def _generate_obo_terms(prefix, ignore_mappings=False, map_to_ns=None):
    filename = os.path.join(indra_resources, '%s.json' % prefix)
    logger.info('Loading %s', filename)
    with open(filename) as file:
        entries = json.load(file)
    terms = []
    for entry in entries:
        terms += terms_from_obo_json_entry(entry, prefix=prefix,
                                           ignore_mappings=ignore_mappings,
                                           map_to_ns=map_to_ns)
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


def terms_from_obo_url(url, prefix, ignore_mappings=False, map_to_ns=None):
    """Return terms extracted directly from an OBO given as a URL."""
    import obonet
    from indra.databases.obo_client import OboClient
    g = obonet.read_obo(url)
    entries = OboClient.entries_from_graph(g, prefix=prefix)
    terms = []
    for entry in entries:
        terms += terms_from_obo_json_entry(entry, prefix=prefix,
                                           ignore_mappings=ignore_mappings,
                                           map_to_ns=map_to_ns)
    return terms


def get_all_terms():
    terms = []

    generated_term_groups = [
        generate_uniprot_terms(),
        generate_famplex_terms(),
        generate_hgnc_terms(),
        generate_chebi_terms(),
        generate_go_terms(),
        generate_mesh_terms(),
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
    from .resources import GROUNDING_TERMS_PATH as fname
    terms = get_all_terms()
    dump_terms(terms, fname)


if __name__ == '__main__':
    main()
