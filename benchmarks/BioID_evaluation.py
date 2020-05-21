import os
import json
import argparse
import pandas as pd
import networkx as nx
import lxml.etree as etree
from obonet import read_obo
from datetime import datetime
from collections import defaultdict

from indra.databases.mesh_client import mesh_isa
from indra.databases.uniprot_client import get_hgnc_id
from indra.databases.hgnc_client import get_hgnc_from_entrez
from indra.databases.chebi_client import get_chebi_id_from_pubchem

from gilda.grounder import logger, Grounder

logger.setLevel('WARNING')


class BioIDBenchmarker(object):
    """Used for evaluating gilda using data from BioCreative VI BioID track

    Parameters
    ----------
    bioid_data_path : str
        Path to dataset from BioCreative VI BioID track training data
        It should contain a file annotations.csv which contains columns
        text, obj, and 'don article' (among others) for entity text, curated
        grounding, and an identifier for the paper where the entity was
        tagged respectively. It should also contain a directory fulltext_bioc
        containing xml files for fulltexts of papers used in corpus.
    grounder : Optional[py:class:`gilda.grounder.Grounder]
        Grounder object to use in evaluation. If None, instantiates a grounder
        with default arguments. Default: None
    equivalences : Optional[dict]
        Dictionary of mappings between namespaces. Maps strings of the form
        f'{namespace}:{id}' to strings for equivalent groundings. This is
        used to map groundings from namespaces used the the BioID track
        (e.g. Uberon, Cell Ontology, Cellosaurus, NCBI Taxonomy) that are not
        available by default in Gilda. Default: None
    isa_relations : Optional[dict]
        Dictionary mapping strings of the form f'{namespace}:{id}' to other
        such strings such that if y = isa_relations[x] then x isa y holds.
        Users have the option of considering a Gilda grounding x to match a
        gold standard grounding y if x isa y or y isa x. Default: None
    godag : Optional[py:class:`networkx.MultiDiGraph`]
        Networkx graph of go network taken from file go.obo
    """
    def __init__(self, bioid_data_path, grounder=None,
                 equivalences=None, isa_relations=None, godag=None):
        if grounder is None:
            grounder = Grounder()
        if equivalences is None:
            equivalences = {}
        if isa_relations is None:
            isa_relations = {}
        available_namespaces = set()
        for terms in grounder.entries.values():
            for term in terms:
                available_namespaces.add(term.db)
        self.grounder = grounder
        self.equivalences = equivalences
        self.isa_relations = isa_relations
        self.available_namespaces = list(available_namespaces)
        self.bioid_data_path = bioid_data_path
        self.processed_data = self._process_annotations_table()
        self.godag = godag

    def get_mappings_tables(self):
        """Get table showing how goldstandard groundings are being mapped

        Namespaces used in the Bioc dataset may only partially overlap with
        those used by Gilda. Users may pass in a dictionary of equivalences
        mapping groundings used in the Bioc dataset to Gilda's namespaces.
        This method generated tables showing how groundings used in the
        dataset project onto Gilda's namespaces through these equivalences.

        Returns
        -------
        mapping_table : py:class`pandas.DataFrame`
        Rows correspond to namespaces used in the Bioc dataset, columns
        to namespaces used in Gilda (automatically populated based on a
        Gilda Grounders entries attribute). There is also a row Total
        containing the sum of values for all other rows. There are columns
        Count, and Total Mapped, showing the total count of entries for
        each row namespace, and the total count of entries that could be
        mapped to a Gilda namespace respectively.

        The same row namespace can be mapped to multiple column namespaces,
        causing values in them Total Mapped column to be less than the sum of
        values of other columns in the same row. Additionally, in some cases
        an entry in the Bioc dataset has multiple curated groundings, causing
        the counts not to add up to the number of entries in the dataset.

        mapping_table_unique : py:class`pandas.DataFrame`
        Similar to mapping table, but counts are given for unique named
        entity groundings, ignoring duplication of groundings between rows
        in the Bioc dataset.
        """
        # Namespaces used in Bioc dataset after standardization
        bioc_nmspaces = ['UP', 'NCBI gene', 'Rfam', 'CHEBI', 'PubChem', 'GO',
                         'CL', 'CVCL', 'UBERON', 'NCBI taxon']
        # Mapping of namespaces to row and column names. Namespaces not
        # included will be used as row and column names unmodifed.
        nmspace_displaynames = {'UP': 'Uniprot', 'NCBI gene': 'Entrez',
                                'PubChem': 'PubChem', 'CL': 'Cell Ontology',
                                'CVCL': 'Cellosaurus', 'UBERON': 'Uberon',
                                'FPLX': 'Famplex'}

        def get_display_name(ns):
            """Gets row/column name associated to a namespace"""
            return nmspace_displaynames[ns] \
                if ns in nmspace_displaynames else ns
        # Build dataframes for storing information. Values will be filled in
        # by looping through rows of the dataset.
        index = [get_display_name(ns) for ns in bioc_nmspaces] + ['Total']
        columns = (['Count'] +
                   [get_display_name(ns) for ns in self.available_namespaces] +
                   ['Total Mapped'])
        mapping_table = pd.DataFrame(index=index, columns=columns)
        mapping_table.fillna(0, inplace=True)
        mapping_table_unique = pd.DataFrame(index=index, columns=columns)
        mapping_table_unique.fillna(0, inplace=True)

        # Maps row namespaces to sets of associated grounding ids
        nmspace_ids = defaultdict(set)
        # Maps row namespaces to to set of Gilda grounding ids that have
        # been mapped to from them
        mapped_to_nmspace_ids = defaultdict(set)
        # Maps row namespaces to sets of associated grounding ids, but
        # only in cases where some mapping exists to a Gilda grounding
        mapped_from_nmspace_ids = defaultdict(set)
        # Looping through dataframe is costly. There may be a way to write
        # this with a clever series of groupbys
        for _, row in self.processed_data.iterrows():
            # For each row loop through goldstandard groundings. There can
            # be more than one
            for g1 in row.obj:
                # Get the namespace. If it is not one of the namespaces used
                # in evaluation, discard and continue to the next iteration
                # of the loop
                nmspace1 = g1.split(':', maxsplit=1)[0]
                if nmspace1 not in bioc_nmspaces:
                    continue
                # Increment total count for this namespace
                mapping_table.loc[get_display_name(nmspace1), 'Count'] += 1
                # If this particular grounding has not been seen before for
                # this namespace increment unique count and mark grounding
                # as having been seen
                if g1 not in nmspace_ids[nmspace1]:
                    mapping_table_unique.loc[get_display_name(nmspace1),
                                             'Count'] += 1
                    nmspace_ids[nmspace1].add(g1)
                # Get all of the synonyms that grounding can be mapped to.
                # This includes the grounding itself. If a row namespace is
                # also a column namespace, we consider this to be a valid
                # mapping
                synonyms = self.get_synonym_set([g1])
                # Track which namespaces have been used so we don't overcount
                # when the same grounding can be mapped to multiple groundings
                # in the same namespace
                used_namespaces = set()
                for g2 in synonyms:
                    nmspace2 = g2.split(':', maxsplit=1)[0]
                    # If a namespace mapped to is not available in Gilda
                    # or if we have already tallied a mapping to this namespace
                    # for this particular row, discard and continue
                    if nmspace2 not in self.available_namespaces or \
                       nmspace2 in used_namespaces:
                        continue
                    # If Gilda namespace has not been mapped to in the curent
                    # row increment the count of entries in the namespace with
                    # a mapping to a Gilda namespace
                    if not used_namespaces:
                        mapping_table.loc[get_display_name(nmspace1),
                                          'Total Mapped'] += 1
                    used_namespaces.add(nmspace2)
                    # If the grounding g1 has never been mapped to a Gilda
                    # namespace increment the unique count
                    if g1 not in mapped_from_nmspace_ids[nmspace1]:
                        mapping_table_unique.\
                            loc[get_display_name(nmspace1),
                                'Total Mapped'] += 1
                        mapped_from_nmspace_ids[nmspace1].add(g1)
                    # Increment count for mapping of row namespace to
                    # column namespace
                    mapping_table.loc[get_display_name(nmspace1),
                                      get_display_name(nmspace2)] += 1
                    # If the grounding in column namespace has not been mapped
                    # to by the grounding in row namespace, increment unique
                    # count
                    if g2 not in mapped_to_nmspace_ids[nmspace1]:
                        mapping_table_unique.\
                            loc[get_display_name(nmspace1),
                                get_display_name(nmspace2)] += 1
                        mapped_to_nmspace_ids[nmspace1].add(g2)
        # Generate total rows
        mapping_table.loc['Total', :] = mapping_table.sum()
        mapping_table_unique.loc['Total', :] = mapping_table_unique.sum()
        mapping_table.reset_index(inplace=True)
        mapping_table.rename({'index': 'Namespace'}, inplace=True)
        mapping_table_unique.reset_index(inplace=True)
        mapping_table_unique.rename({'index': 'Namespace'}, inplace=True)
        return mapping_table, mapping_table_unique

    def _process_annotations_table(self):
        """Extract relevant information from annotations table."""
        df = pd.read_csv(os.path.join(self.bioid_data_path,
                                      'annotations.csv'),
                         sep=',', low_memory=False)
        # Split entries with multiple groundings
        df.loc[:, 'obj'] = df['obj'].\
            apply(lambda x: x.split('|'))
        # Normalize ids
        df.loc[:, 'obj'] = df['obj'].\
            apply(lambda x: [self._normalize_id(y) for y in x])
        # Add synonyms of gold standard groundings to help match more things
        df.loc[:, 'obj_synonyms'] = df['obj'].\
            apply(lambda x: self.get_synonym_set(x))
        # Create column for entity type
        df.loc[:, 'entity_type'] = df.\
            apply(lambda row: self._get_entity_type(row.obj)
                  if self._get_entity_type(row.obj) != 'Gene' else
                  'Human Gene' if any([y.startswith('HGNC') for y in
                                       row.obj_synonyms]) else
                  'Nonhuman Gene', axis=1)
        processed_data = df[['text', 'obj', 'obj_synonyms', 'entity_type',
                             'don_article']]
        processed_data = processed_data[processed_data.entity_type
                                        != 'unknown']
        return processed_data

    def ground_entities_with_gilda(self):
        """Compute gilda groundings of entity texts in corpus

        Adds two columns to the internal dataframe for groundings with
        and without context based disambiguation.
        """
        df = self.processed_data
        df.loc[:, 'groundings_no_context'] = df.text.\
            apply(lambda x: self._get_grounding_list(x))
        df.loc[:, 'groundings'] = df.\
            apply(lambda row:
                  self._get_grounding_list(
                      row.text,
                      context=self._get_plaintext(row.don_article)), axis=1)
        self._evaluate_gilda_performance()

    def _get_plaintext(self, don_article):
        """Get plaintext content from XML file in BioID corpus

        Parameters
        ----------
        don_article : str
            Identifier for paper used within corpus.

        Returns
        -------
        str
            Plaintext of specified article
        """
        tree = etree.parse(os.path.join(self.bioid_data_path,
                                        'fulltext_bioc',
                                        f'{don_article}.xml'))
        paragraphs = tree.xpath('//text')
        paragraphs = [' '.join(text.itertext()) for text in paragraphs]
        return '/n'.join(paragraphs) + '/n'

    def _normalize_id(self, id_):
        """Convert ID into standardized format, f'{namespace}:{id}'."""
        if id_.startswith('CVCL'):
            return id_.replace('_', ':')
        split_id = id_.split(':', maxsplit=1)
        if split_id[0] == 'Uberon':
            return split_id[1]
        if split_id[0] == 'Uniprot':
            return f'UP:{split_id[1]}'
        if split_id[0] in ['GO', 'CHEBI']:
            return f'{split_id[0]}:{split_id[0]}:{split_id[1]}'
        return id_

    def _get_entity_type(self, bioc_groundings):
        """Get entity type based on entity groundings of text in corpus.
        """
        if any([x.startswith('NCBI gene')
                or x.startswith('UP') for x in bioc_groundings]):
            result = 'Gene'
        elif any([x.startswith('Rfam') for x in bioc_groundings]):
            result = 'miRNA'
        elif any([x.startswith('CHEBI') or x.startswith('PubChem')
                  for x in bioc_groundings]):
            result = 'Small Molecule'
        elif any([x.startswith('GO') for x in bioc_groundings]):
            result = 'Cellular Component'
        elif any([x.startswith('CVCL') or x.startswith('CL')
                  for x in bioc_groundings]):
            result = 'Cell types/Cell lines'
        elif any([x.startswith('UBERON') for x in bioc_groundings]):
            result = 'Tissue/Organ'
        elif any([x.startswith('NCBI taxon') for x in bioc_groundings]):
            result = 'Taxon'
        else:
            result = 'unknown'
        return result

    def _get_grounding_list(self, text, context=None):
        """Run gilda on a text and extract list of result-score tuples."""
        groundings = self.grounder.ground(text, context=context)
        result = []
        for grounding in groundings:
            db, id_ = grounding.term.db, grounding.term.id
            result.append((f'{db}:{id_}', grounding.score))
        return result

    def get_synonym_set(self, grounding_list):
        """Return set containing all elements in input list along with synonyms
        """
        output = set()
        for id_ in grounding_list:
            output.update(self._get_equivalent_entities(id_))
        return output

    def _get_equivalent_entities(self, id_):
        """Return set of equivalent entity groundings

        Uses set of equivalences in self.equiv_map as well as those
        available in indra's hgnc, uniprot, and chebi clients.
        """
        output = set([id_])
        db, value = id_.split(':', maxsplit=1)
        if id_ in self.equivalences:
            output.update(self.equivalences[id_])
        hgnc_id = None
        if db == 'NCBI gene':
            hgnc_id = get_hgnc_from_entrez(value)
        if db == 'UP':
            hgnc_id = get_hgnc_id(value)
        if hgnc_id is not None:
            output.add(f'HGNC:{hgnc_id}')
        if db == 'PubChem':
            chebi_id = get_chebi_id_from_pubchem(value)
            if chebi_id is not None:
                output.add(f'CHEBI:CHEBI:{chebi_id}')
        return output

    def famplex_isa(self, hgnc_id, fplx_id):
        """Check if hgnc entity satisfies and isa relation with famplex entity

        Parameters
        ----------
        hgnc_id : str
            String of the form f'{namespace}:{id}'
        fplx_id : str
            String of the form f'{namespace}:{id}'

        Returns
        -------
        bool
            True if hgnc_id corresponds to a valid HGNC grounding,
            fplx_id corresponds to a valid Famplex grounding and the
            former isa the later.
        """
        return hgnc_id in self.isa_relations and \
            fplx_id in self.isa_relations[hgnc_id]

    def isa(self, id1, id2):
        """True if id1 satisfies isa relationship with id2

        Is aware of MESH, GO, and any isa relationships provided in
        isa_relations dict. At this time only looks at parents and children
        in isa_relations dict, does not follow paths.
        """
        if id1.startswith('MESH') and id2.startswith('MESH'):
            return mesh_isa(id1, id2)
        elif id1.startswith('GO') and id2.startswith('GO'):
            id1 = id1.split(':', maxsplit=1)[1]
            id2 = id2.split(':', maxsplit=1)[1]
            try:
                return nx.has_path(self.godag, id1, id2)
            except Exception:
                return False
        else:
            return id1 in self.isa_relations and id2 in self.isa_relations[id1]

    def _evaluate_gilda_performance(self):
        """Calculate statistics showing Gilda's performance on corpus

        Directly updates internal dataframe
        """
        def top_correct(row, disamb=True):
            groundings = row.groundings if disamb \
                else row.groundings_no_context
            if not groundings:
                return False
            groundings = [g[0] for g in groundings]
            top_grounding = groundings[0]
            return set([top_grounding]) <= set(row.obj_synonyms)

        def exists_correct(row, disamb=True):
            groundings = row.groundings if disamb \
                else row.groundings_no_context
            if not groundings:
                return False
            groundings = [g[0] for g in groundings]
            return len(set(groundings) & set(row.obj_synonyms)) > 0

        def top_correct_w_fplx(row, disamb=True):
            groundings = row.groundings if disamb \
                else row.groundings_no_context
            if not groundings:
                return False
            groundings = [g[0] for g in groundings]
            top_grounding = groundings[0]
            return any([x == top_grounding or
                        self.famplex_isa(x, top_correct)
                        for x in row.obj_synonyms])

        def exists_correct_w_fplx(row, disamb=True):
            groundings = row.groundings if disamb \
                else row.groundings_no_context
            if not groundings:
                return False
            groundings = [g[0] for g in groundings]
            return any([x == y or self.famplex_isa(x, y)
                        for x in row.obj_synonyms
                        for y in groundings])

        def top_correct_loose(row, disamb=True):
            groundings = row.groundings if disamb \
                else row.groundings_no_context
            if not groundings:
                return False
            groundings = [g[0] for g in groundings]
            top_grounding = groundings[0]
            return any([x == top_grounding or
                        self.isa(x, top_grounding) or
                        self.isa(top_grounding, x)
                        for x in row.obj_synonyms])

        def exists_correct_loose(row, disamb=True):
            groundings = row.groundings if disamb \
                else row.groundings_no_context
            if not groundings:
                return False
            groundings = [g[0] for g in groundings]
            return any([x == y or
                        self.isa(x, y) or
                        self.isa(y, x)
                        for x in row.obj_synonyms
                        for y in groundings])

        df = self.processed_data
        df.loc[:, 'top_correct'] = df.apply(top_correct, axis=1)
        df.loc[:, 'top_correct_w_fplx'] = df.apply(top_correct_w_fplx, axis=1)
        df.loc[:, 'top_correct_loose'] = df.apply(top_correct_loose, axis=1)
        df.loc[:, 'exists_correct'] = df.apply(exists_correct, axis=1)
        df.loc[:, 'exists_correct_w_fplx'] = df.\
            apply(exists_correct_w_fplx, axis=1)
        df.loc[:, 'exists_correct_loose'] = df.\
            apply(exists_correct_loose, axis=1)
        df.loc[:, 'top_correct_no_context'] = df.\
            apply(lambda row: top_correct(row, False), axis=1)
        df.loc[:, 'top_correct_w_fplx_no_context'] = df.\
            apply(lambda row: top_correct_w_fplx(row, False), axis=1)
        df.loc[:, 'top_correct_loose_no_context'] = df.\
            apply(lambda row: top_correct_loose(row, False), axis=1)
        df.loc[:, 'Has Grounding'] = df.groundings.\
            apply(lambda x: len(x) > 0)

    def get_results_tables(self, match='loose', with_context=True):
        """Get tables of results

        Parameters
        ----------
        match : Optional[str]
            One of 'strict', 'w_fplex', or 'loose'. 'strict' only counts
            a Gilda grounding as a match if it is an exact match or equivalent
            to at least one of the curated groundings for the entry
            (some entries have multiple equivalent curated groundings).
            'w_fplex' also counts a Gilda grounding as a match if the curated
            grounding has an HGNC equivalent and satisfies and isa relationship
            with the Gilda grounding. 'loose' counts the pair x, y of a curated
            grounding and a Gilda groundings as a match if x isa y or y isa x
            within the FPLX, MESH, and GO ontologies, or within the dictionary
            of isa_relations.

        Returns
        -------
        counts_table : py:class:`pandas.DataFrame`
        precision_recall : py:class:`pandas.DataFrame`
        disamb_table : py:class:`pandas.DataFrame`
        """
        if match not in ['strict', 'w_fplex', 'loose']:
            raise ValueError("match must be one of 'strict', 'w_famplex', or"
                             " 'loose'.")
        df = self.processed_data
        if 'top_correct' not in df.columns:
            raise RuntimeError('Gilda groundings have not been computed')
        res_df = df[['entity_type', 'top_correct', 'top_correct_no_context',
                     'exists_correct', 'top_correct_w_fplx',
                     'top_correct_w_fplx_no_context', 'exists_correct_w_fplx',
                     'top_correct_loose', 'top_correct_loose_no_context',
                     'exists_correct_loose', 'Has Grounding']].copy()
        res_df.loc[:, 'Total'] = True
        total = res_df.drop('entity_type', axis=1).sum()
        total = total.to_frame().transpose()
        total.loc[:, 'entity_type'] = 'Total'
        stats = res_df.groupby('entity_type', as_index=False).sum()
        stats = stats[stats['entity_type'] != 'unknown']
        stats = stats.append(total, ignore_index=True)
        stats.loc[:, stats.columns[1:]] = stats[stats.columns[1:]].astype(int)
        if match == 'strict':
            score_cols = ['top_correct', 'exists_correct']
        else:
            score_cols = [f'top_correct_{match}', f'exists_correct_{match}']
        if not with_context:
            score_cols[0] = score_cols[0] + ['_no_context']
        cols = ['entity_type'] + score_cols + ['Has Grounding', 'Total']
        counts_table = stats[cols]
        new_column_names = ['Entity Type', 'Correct', 'Exists Correct',
                            'Has Grounding', 'Total']
        counts_table.columns = new_column_names
        precision_recall = pd.DataFrame(index=stats.index,
                                        columns=['Entity Type',
                                                 'Precision',
                                                 'Exists Correct PR',
                                                 'Recall',
                                                 'Exists Correct RC'])
        precision_recall.loc[:, 'Entity Type'] = counts_table['Entity Type']
        precision_recall.loc[:, 'Precision'] = \
            round(counts_table['Correct'] /
                  counts_table['Has Grounding'], 3)
        precision_recall.loc[:, 'Exists Correct PR'] = \
            round(counts_table['Exists Correct'] /
                  counts_table['Has Grounding'], 3)
        precision_recall.loc[:, 'Recall'] = round(counts_table['Correct'] /
                                                  counts_table['Total'], 3)
        precision_recall.loc[:, 'Exists Correct RC'] = \
            round(counts_table['Exists Correct'] / counts_table['Total'], 3)
        cols = ['entity_type', 'top_correct_loose_no_context',
                'top_correct_loose', 'exists_correct', 'Total']
        new_column_names = ['Entity Type', 'Correct', 'Correct (disamb)',
                            'Exists Correct', 'Total']
        disamb_table = stats[cols]
        disamb_table.columns = new_column_names
        return counts_table, precision_recall, disamb_table


def make_table_printable(df):
    """Return table in printable format

    Output to markdown if tabulate is installed, otherwise just convert
    to string.
    """
    try:
        output = df.to_markdown(showindex=False)
    except ImportError:
        output = df.to_string(index=False)
    return output


if __name__ == '__main__':
    """Run this script to evaluate gilda on the BioCreative VI BioID corpus.

    It has two optional arguments, --datapath and --resultspath that specify
    the path to the directory with necessary data and the path to the
    directory where results will be stored. Results files will be added to
    the results directory in timestamped files.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Benchmark gilda on BioID'
                                     ' corpus.')
    parser.add_argument('--datapath', type=str,
                        default=os.path.join(path, 'data'))
    parser.add_argument('--resultspath', type=str,
                        default=os.path.join(path, 'results'))
    args = parser.parse_args()
    data_path = args.datapath
    data_path = os.path.expandvars(os.path.expanduser(data_path))
    results_path = args.resultspath
    results_path = os.path.expandvars(os.path.expanduser(results_path))
    try:
        with open(os.path.join(data_path, 'equivalences.json')) as f:
            equivalences = json.load(f)
    except FileNotFoundError:
        equivalences = {}
    try:
        with open(os.path.join(data_path, 'isa_relations.json')) as f:
            isa_relations = json.load(f)
    except FileNotFoundError:
        isa_relations = {}
    godag = read_obo('data/go.obo')
    benchmarker = BioIDBenchmarker(os.path.join(data_path, 'BioIDtraining_2'),
                                   equivalences=equivalences,
                                   isa_relations=isa_relations,
                                   godag=godag)
    benchmarker.ground_entities_with_gilda()
    mappings_table, mappings_table_unique = benchmarker.get_mappings_tables()
    counts, precision_recall, disamb_table = benchmarker.get_results_tables()
    try:
        _ = precision_recall.to_markdown()
    except ImportError:
        print("Install tabulate with `pip install tabulate` to pretty print"
              " table output.")
    print(make_table_printable(precision_recall))
    # Generate output document
    caption1 = """
    Table 1:
    Mapping of groundings for entities in BioID corpus into namespaces used by
    Gilda. Count is by entries in corpus with groundings being counted multiple
    times if they occur in more than one entry."""
    table1 = make_table_printable(mappings_table)
    caption2 = """
    Table 2:
    Mapping of groundings for entities in BioID corpus into Namespaces used by
    Gilda. Count is by unique groundings, with the same grounding only being
    counted once even if it appears in many entries."""
    table2 = make_table_printable(mappings_table_unique)
    caption3 = """
    Table 3:
    Counts of number of entries in corpus for each entity type, along with
    number of entries where Gilda's top grounding is correct, the number
    where one of Gilda's groundings is correct, and the number of entries
    where Gilda produced some grounding. Context based disambiguation is
    applied and Gilda's groundings are considered correct if there is
    an isa relation between the goldstandard grounding and Gilda's or
    vice versa."""
    table3 = make_table_printable(counts)
    caption4 = """
    Table 4:
    Precision and recall values for Gilda performance by entity type. Values
    are given both for the case where Gilda is considered correct only if the
    top grounding matches and the case where Gilda is considered correct if
    any of its groundings match."""
    table4 = make_table_printable(precision_recall)
    caption5 = """
    Table 5:
    Comparison of results with and without context based disambiguation."""
    table5 = make_table_printable(disamb_table)
    output = '\n'.join([caption1, table1, caption2, table2,
                        caption3, table3, caption4, table4,
                        caption5, table5])
    time = datetime.now().strftime('%y-%m-%d-%H:%M:%S')
    outname = f'benchmark_{time}'
    with open(os.path.join(results_path, outname), 'w') as f:
        f.write(output)
