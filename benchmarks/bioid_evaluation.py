"""This script benchmarks Gilda on the BioCreative VII BioID corpus.
It dumps multiple result tables in the results folder."""
import json
import os
import pathlib
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import lru_cache
from textwrap import dedent
from typing import Any, Collection, Dict, Iterable, List, Optional, Set, Tuple

import click
import pandas as pd
import pystow
import tabulate
from lxml import etree
from tqdm import tqdm

import famplex
from gilda import __version__
from gilda.grounder import Grounder, logger
from gilda.resources import mesh_to_taxonomy, popular_organisms
from indra.databases.chebi_client import get_chebi_id_from_pubchem
from indra.databases.hgnc_client import get_hgnc_from_entrez
from indra.databases.uniprot_client import get_hgnc_id
from indra.literature import pmc_client, pubmed_client
from indra.ontology.bio import bio_ontology

logger.setLevel('WARNING')

HERE = os.path.dirname(os.path.abspath(__file__))
TAXONOMY_CACHE_PATH = os.path.join(HERE, 'taxonomy_cache.json')
MODULE = pystow.module('gilda', 'biocreative')
URL = 'https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz'

tqdm.pandas()

#: A set of the prefix->prefix mappings missing from the bio-ontology
BO_MISSING_XREFS = set()



class BioIDBenchmarker:
    """Used for evaluating gilda using data from BioCreative VI BioID track

    Parameters
    ----------
    grounder :
        Grounder object to use in evaluation. If None, instantiates a grounder
        with default arguments. Default: None
    equivalences :
        Dictionary of mappings between namespaces. Maps strings of the form
        f'{namespace}:{id}' to strings for equivalent groundings. This is
        used to map groundings from namespaces used the the BioID track
        (e.g. Uberon, Cell Ontology, Cellosaurus, NCBI Taxonomy) that are not
        available by default in Gilda. Default: None
    """

    def __init__(
        self,
        *,
        grounder: Optional[Grounder] = None,
        equivalences: Optional[Dict[str, Any]] = None,
    ):
        print("Instantiating benchmarker...")
        if grounder is None:
            grounder = Grounder()
        print("Instantiating bio ontology...")
        bio_ontology.initialize()
        if equivalences is None:
            equivalences = {}
        available_namespaces = set()
        for terms in grounder.entries.values():
            for term in terms:
                available_namespaces.add(term.db)
        self.grounder = grounder
        self.equivalences = equivalences
        self.available_namespaces = list(available_namespaces)
        self.paper_level_grounding = defaultdict(set)
        self.processed_data = self._process_annotations_table()
        if os.path.exists(TAXONOMY_CACHE_PATH):
            with open(TAXONOMY_CACHE_PATH, 'r') as fh:
                self.taxonomy_cache = json.load(fh)
        else:
            self.taxonomy_cache = {}
        print('Taxonomy cache length: %s' % len(self.taxonomy_cache))

    def get_mappings_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                        mapping_table_unique. \
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
                        mapping_table_unique. \
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
        print("Extracting information from annotations table...")
        df = MODULE.ensure_tar_df(
            url=URL,
            inner_path='BioIDtraining_2/annotations.csv',
            read_csv_kwargs=dict(sep=',', low_memory=False),
        )
        # Split entries with multiple groundings then normalize ids
        df.loc[:, 'obj'] = df['obj'].apply(self._normalize_ids)
        # Add synonyms of gold standard groundings to help match more things
        df.loc[:, 'obj_synonyms'] = df['obj'].apply(self.get_synonym_set)
        # Create column for entity type
        df.loc[:, 'entity_type'] = df.apply(self._get_entity_type_helper, axis=1)
        processed_data = df[['text', 'obj', 'obj_synonyms', 'entity_type',
                             'don_article']]
        print("%d rows in processed annotations table." % len(processed_data))
        processed_data = processed_data[processed_data.entity_type
                                        != 'unknown']
        print("%d rows in annotations table without unknowns." %
              len(processed_data))
        for don_article, text, synonyms in df[['don_article', 'text',
                                               'obj_synonyms']].values:
            self.paper_level_grounding[don_article, text].update(synonyms)
        return processed_data

    def _get_entity_type_helper(self, row) -> str:
        if self._get_entity_type(row.obj) != 'Gene':
            return self._get_entity_type(row.obj)
        elif any(y.startswith('HGNC') for y in row.obj_synonyms):
            return 'Human Gene'
        else:
            return 'Nonhuman Gene'

    def ground_entities_with_gilda(self, context=True, species=True):
        """Compute gilda groundings of entity texts in corpus

        Adds two columns to the internal dataframe for groundings with
        and without context based disambiguation.
        """
        df = self.processed_data
        tqdm.write("Grounding no-context corpus with Gilda...")
        df.loc[:, 'groundings_no_context'] = df.text. \
            progress_apply(self._get_grounding_list)

        if context and species:
            tqdm.write("Grounding with-context corpus with Gilda...")
            # use from tqdm.contrib.concurrent import thread_map
            df.loc[:, 'groundings'] = df. \
                progress_apply(self._get_row_grounding_list_with_model, axis=1)
        elif not context and species:
            df.loc[:, 'groundings'] = df. \
                progress_apply(self._get_row_grounding_list_sans_model, axis=1)
        elif context and not species:
            raise NotImplementedError
        else:
            raise ValueError("why would we ever do this")
            # tqdm.write("Skipping grounding with context.")
            # df.loc[:, 'groundings'] = df.groundings_no_context
        tqdm.write("Finished grounding corpus with Gilda...")
        self._evaluate_gilda_performance()

    def _get_row_grounding_list_with_model(self, row):
        return self._get_grounding_list(
            row.text,
            context=self._get_plaintext(row.don_article),
            organisms=self._get_organism_priority(row.don_article),
        )

    def _get_row_grounding_list_sans_model(self, row):
        return self._get_grounding_list(
            row.text,
            organisms=self._get_organism_priority(row.don_article),
        )

    @lru_cache(maxsize=None)
    def _get_plaintext(self, don_article: str) -> str:
        """Get plaintext content from XML file in BioID corpus

        Parameters
        ----------
        don_article :
            Identifier for paper used within corpus.

        Returns
        -------
        :
            Plaintext of specified article
        """
        directory = MODULE.ensure_untar(url=URL, directory='BioIDtraining_2')
        path = directory.joinpath('BioIDtraining_2', 'fulltext_bioc',
                                  f'{don_article}.xml')
        tree = etree.parse(path.as_posix())
        paragraphs = tree.xpath('//text')
        paragraphs = [' '.join(text.itertext()) for text in paragraphs]
        return '/n'.join(paragraphs) + '/n'

    def _get_organism_priority(self, don_article):
        don_article = str(don_article)
        if don_article in self.taxonomy_cache:
            return self.taxonomy_cache[don_article]
        pubmed_id = pubmed_from_pmc(don_article)
        taxonomy_ids = get_taxonomy_for_pmid(pubmed_id)
        organisms = [o for o in popular_organisms
                     if o in taxonomy_ids] + \
                    [o for o in popular_organisms
                     if o not in taxonomy_ids]
        self.taxonomy_cache[don_article] = organisms
        return organisms

    @classmethod
    def _normalize_ids(cls, curies: str) -> List[str]:
        return [cls._normalize_id(y) for y in curies.split('|')]

    @staticmethod
    def _normalize_id(curie):
        """Convert ID into standardized format, f'{namespace}:{id}'."""
        if curie.startswith('CVCL'):
            return curie.replace('_', ':')
        split_id = curie.split(':', maxsplit=1)
        if split_id[0] == 'Uberon':
            return split_id[1]
        if split_id[0] == 'Uniprot':
            return f'UP:{split_id[1]}'
        if split_id[0] in ['GO', 'CHEBI']:
            return f'{split_id[0]}:{split_id[0]}:{split_id[1]}'
        return curie

    @staticmethod
    def _get_entity_type(groundings: Collection[str]) -> str:
        """Get entity type based on entity groundings of text in corpus."""
        if any(
            grounding.startswith('NCBI gene') or grounding.startswith('UP')
            for grounding in groundings
        ):
            return 'Gene'
        elif any(grounding.startswith('Rfam') for grounding in groundings):
            return 'miRNA'
        elif any(grounding.startswith('CHEBI') or grounding.startswith('PubChem')
                 for grounding in groundings):
            return 'Small Molecule'
        elif any(grounding.startswith('GO') for grounding in groundings):
            return 'Cellular Component'
        elif any(
            grounding.startswith('CVCL') or grounding.startswith('CL')
            for grounding in groundings
        ):
            return 'Cell types/Cell lines'
        elif any(grounding.startswith('UBERON') for grounding in groundings):
            return 'Tissue/Organ'
        elif any(grounding.startswith('NCBI taxon') for grounding in groundings):
            return 'Taxon'
        else:
            return 'unknown'

    def _get_grounding_list(
        self,
        text: str,
        context=None,
        organisms=None,
    ) -> List[Tuple[str, float]]:
        """Run gilda on a text and extract list of result-score tuples."""
        groundings = self.grounder.ground(text, context=context,
                                          organisms=organisms)
        result = []
        for grounding in groundings:
            db, id_ = grounding.term.db, grounding.term.id
            result.append((f'{db}:{id_}', grounding.score))
        return result

    def get_synonym_set(self, curies: Iterable[str]) -> Set[str]:
        """Return set containing all elements in input list along with synonyms
        """
        output = set()
        for curie in curies:
            output.update(self._get_equivalent_entities(curie))
        # We accept all FamPlex terms that cover some or all of the specific
        # entries in the annotations
        covered_fplx = {fplx_entry for fplx_entry, members
                        in fplx_members.items() if (members <= output)}
        output |= {'FPLX:%s' % fplx_entry for fplx_entry in covered_fplx}
        return output

    def _get_equivalent_entities(self, curie: str) -> Set[str]:
        """Return set of equivalent entity groundings

        Uses set of equivalences in self.equiv_map as well as those
        available in indra's hgnc, uniprot, and chebi clients.
        """
        output = {curie}
        prefix, identifier = curie.split(':', maxsplit=1)
        for xref_prefix, xref_id in bio_ontology.get_mappings(prefix, identifier):
            output.add(f'{xref_prefix}:{xref_id}')

        # TODO these should all be in bioontology, eventually
        for xref_curie in self.equivalences.get(curie, []):
            if xref_curie in output:
                continue
            xref_prefix, xref_id = xref_curie.split(':', maxsplit=1)
            if (prefix, xref_prefix) not in BO_MISSING_XREFS:
                BO_MISSING_XREFS.add((prefix, xref_prefix))
                tqdm.write(f'Bioontology v{bio_ontology.version} is missing mappings from {prefix} to {xref_prefix}')
            output.add(xref_curie)

        if prefix == 'NCBI gene':
            hgnc_id = get_hgnc_from_entrez(identifier)
            if hgnc_id is not None:
                output.add(f'HGNC:{hgnc_id}')
        if prefix == 'UP':
            hgnc_id = get_hgnc_id(identifier)
            if hgnc_id is not None:
                output.add(f'HGNC:{hgnc_id}')
        if prefix == 'PubChem':
            chebi_id = get_chebi_id_from_pubchem(identifier)
            if chebi_id is not None:
                output.add(f'CHEBI:{chebi_id}')
        return output

    @staticmethod
    def famplex_isa(hgnc_id: str, fplx_id: str) -> bool:
        """Check if hgnc entity satisfies and isa relation with famplex entity

        Parameters
        ----------
        hgnc_id :
            String of the form f'{namespace}:{id}'
        fplx_id :
            String of the form f'{namespace}:{id}'

        Returns
        -------
        :
            True if hgnc_id corresponds to a valid HGNC grounding,
            fplx_id corresponds to a valid Famplex grounding and the
            former isa the later.
        """
        # TODO can this be swapped directly for the bioontology?
        return famplex.isa('HGNC', hgnc_id, 'FPLX', fplx_id)

    def isa(self, curie_1: str, curie_2: str) -> bool:
        """True if id1 satisfies isa relationship with id2."""
        # if curie_1.startswith('MESH') and curie_2.startswith('MESH'):
        #     return mesh_isa(curie_1, curie_2)
        # # Handle GOGO problem
        # elif curie_1.startswith('GO') and curie_2.startswith('GO'):
        #     curie_1 = curie_1.split(':', maxsplit=1)[1]
        #     curie_2 = curie_2.split(':', maxsplit=1)[1]
        #     try:
        #         return nx.has_path(self.godag, curie_1, curie_2)
        #     except Exception:
        #         return False
        # else:
        #     return curie_1 in self.isa_relations and curie_2 in self.isa_relations[curie_1]
        ns1, id1 = curie_1.split(':', maxsplit=1)
        ns2, id2 = curie_2.split(':', maxsplit=1)
        # TODO did we need to keep some processing on the IDs?
        return bio_ontology.isa(ns1, id1, ns2, id2)

    def top_correct(self, row, disamb=True) -> bool:
        groundings = row.groundings if disamb \
            else row.groundings_no_context
        if not groundings:
            return False
        groundings = [g[0] for g in groundings]
        top_grounding = groundings[0]
        ref_groundings = \
            self.paper_level_grounding[(row.don_article, row.text)]
        return top_grounding in ref_groundings

    def exists_correct(self, row, disamb: bool = True) -> bool:
        groundings = row.groundings if disamb \
            else row.groundings_no_context
        if not groundings:
            return False
        groundings = {g[0] for g in groundings}
        ref_groundings = \
            self.paper_level_grounding[(row.don_article, row.text)]
        return len(groundings & ref_groundings) > 0

    def top_correct_w_fplx(self, row, disamb: bool = True) -> bool:
        groundings = row.groundings if disamb \
            else row.groundings_no_context
        if not groundings:
            return False
        groundings = [g[0] for g in groundings]
        top_grounding = groundings[0]
        ref_groundings = \
            self.paper_level_grounding[(row.don_article, row.text)]
        return any(
            x == top_grounding or self.famplex_isa(x, top_grounding)
            for x in ref_groundings
        )

    def exists_correct_w_fplx(self, row, disamb: bool = True) -> bool:
        groundings = row.groundings if disamb \
            else row.groundings_no_context
        if not groundings:
            return False
        groundings = [g[0] for g in groundings]
        ref_groundings = \
            self.paper_level_grounding[(row.don_article, row.text)]
        return any(
            x == y or self.famplex_isa(x, y)
            for x in ref_groundings
            for y in groundings
        )

    def top_correct_loose(self, row, disamb=True) -> bool:
        groundings = row.groundings if disamb \
            else row.groundings_no_context
        if not groundings:
            return False
        groundings = [g[0] for g in groundings]
        top_grounding = groundings[0]
        ref_groundings = \
            self.paper_level_grounding[(row.don_article, row.text)]
        return any(
            x == top_grounding
            or self.isa(x, top_grounding)
            or self.isa(top_grounding, x)
            for x in ref_groundings
        )

    def exists_correct_loose(self, row, disamb=True) -> bool:
        groundings = row.groundings if disamb \
            else row.groundings_no_context
        if not groundings:
            return False
        groundings = [g[0] for g in groundings]
        ref_groundings = \
            self.paper_level_grounding[(row.don_article, row.text)]
        return any(
            x == y
            or self.isa(x, y)
            or self.isa(y, x)
            for x in ref_groundings
            for y in groundings
        )

    def _evaluate_gilda_performance(self):
        """Calculate statistics showing Gilda's performance on corpus

        Directly updates internal dataframe
        """
        print("Evaluating performance...")
        df = self.processed_data
        df.loc[:, 'top_correct'] = df.apply(self.top_correct, axis=1)
        df.loc[:, 'top_correct_w_fplx'] = df.apply(self.top_correct_w_fplx, axis=1)
        df.loc[:, 'top_correct_loose'] = df.apply(self.top_correct_loose, axis=1)
        df.loc[:, 'exists_correct'] = df.apply(self.exists_correct, axis=1)
        df.loc[:, 'exists_correct_w_fplx'] = df. \
            apply(self.exists_correct_w_fplx, axis=1)
        df.loc[:, 'exists_correct_loose'] = df. \
            apply(self.exists_correct_loose, axis=1)
        df.loc[:, 'top_correct_no_context'] = df. \
            apply(lambda row: self.top_correct(row, False), axis=1)
        df.loc[:, 'top_correct_w_fplx_no_context'] = df. \
            apply(lambda row: self.top_correct_w_fplx(row, False), axis=1)
        df.loc[:, 'top_correct_loose_no_context'] = df. \
            apply(lambda row: self.top_correct_loose(row, False), axis=1)
        df.loc[:, 'Has Grounding'] = df.groundings. \
            apply(lambda x: len(x) > 0)
        print("Finished evaluating performance...")

    def get_results_tables(
        self,
        match: Optional[str] = 'loose',
        with_context: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get tables of results

        Parameters
        ----------
        match :
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
            # OLD
            # score_cols[0] = score_cols[0] + ['_no_context']
            score_cols[0] = score_cols[0] + '_no_context'
        cols = ['entity_type'] + score_cols + ['Has Grounding', 'Total']
        counts_table = deepcopy(stats[cols])
        new_column_names = ['Entity Type', 'Correct', 'Exists Correct',
                            'Has Grounding', 'Total']
        counts_table.columns = new_column_names
        precision_recall = pd.DataFrame(
            index=stats.index,
            columns=[
                'Entity Type',
                'Precision',
                'Exists Correct PR',
                'Recall',
                'Exists Correct RC',
            ],
        )

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

        precision_recall.loc[:, 'Correct F1'] = \
            f1(precision_recall.loc[:, 'Precision'],
               precision_recall.loc[:, 'Recall'])
        precision_recall.loc[:, 'Exists Correct F1'] = \
            f1(precision_recall.loc[:, 'Exists Correct PR'],
               precision_recall.loc[:, 'Exists Correct RC'])

        cols = ['entity_type', 'top_correct_loose_no_context',
                'top_correct_loose', 'exists_correct', 'Total']
        new_column_names = ['Entity Type', 'Correct', 'Correct (disamb)',
                            'Exists Correct', 'Total']
        disamb_table = stats[cols]
        disamb_table.columns = new_column_names
        return counts_table, precision_recall, disamb_table


def get_taxonomy_for_pmid(pmid: str) -> Set[str]:
    if not pmid:
        return set()
    import time
    tqdm.write(f'Looking up annotations for pmid:{pmid}')
    time.sleep(2)
    mesh_annots = pubmed_client.get_mesh_annotations(pmid)
    if mesh_annots is None:
        return set()
    mesh_ids = {annot['mesh'] for annot in mesh_annots}
    taxonomy_ids = set()
    for mesh_id in mesh_ids:
        if mesh_id in mesh_to_taxonomy:
            taxonomy_ids.add(mesh_to_taxonomy[mesh_id])
        mesh_parents = [id for ns, id in
                        bio_ontology.get_parents('MESH', mesh_id)]
        for mesh_parent in mesh_parents:
            if mesh_parent in mesh_to_taxonomy:
                taxonomy_ids.add(mesh_to_taxonomy[mesh_parent])
    tqdm.write('-----')
    tqdm.write('PMID: %s' % pmid)
    for mesh_annot in mesh_annots:
        tqdm.write(mesh_annot['text'])
    tqdm.write('Taxonomy IDs: %s' % taxonomy_ids)
    tqdm.write('-----')
    return taxonomy_ids


def pubmed_from_pmc(pmc_id):
    pmc_id = str(pmc_id)
    if not pmc_id.startswith('PMC'):
        pmc_id = f'PMC{pmc_id}'
    ids = pmc_client.id_lookup(pmc_id, 'pmcid')
    pmid = ids.get('pmid')
    return pmid


#: Namespaces used in Bioc dataset after standardization
bioc_nmspaces = ['UP', 'NCBI gene', 'Rfam', 'CHEBI', 'PubChem', 'GO',
                 'CL', 'CVCL', 'UBERON', 'NCBI taxon']

#: Mapping of namespaces to row and column names. Namespaces not
#: included will be used as row and column names unmodifed.
nmspace_displaynames = {
    'UP': 'Uniprot', 'NCBI gene': 'Entrez',
    'PubChem': 'PubChem', 'CL': 'Cell Ontology',
    'CVCL': 'Cellosaurus', 'UBERON': 'Uberon',
    'FPLX': 'Famplex'
}


def get_famplex_members():
    from indra.databases import hgnc_client
    fplx_entities = famplex.load_entities()
    fplx_children = defaultdict(set)
    for fplx_entity in fplx_entities:
        members = famplex.individual_members('FPLX', fplx_entity)
        for db_ns, db_id in members:
            if db_ns == 'HGNC':
                db_id = hgnc_client.get_current_hgnc_id(db_id)
                if db_id:
                    fplx_children[fplx_entity].add('%s:%s' % (db_ns, db_id))
    return dict(fplx_children)


fplx_members = get_famplex_members()


# NOTE: these mappings are already integrated into equivalences.json
def get_uberon_mesh_mappings():
    import obonet
    from indra.databases import mesh_client
    g = obonet.read_obo('/Users/ben/src/uberon/src/ontology/uberon-edit.obo')
    mappings = {}
    for node, data in g.nodes(data=True):
        xrefs = [x[5:] for x in data.get('xref', []) if x.startswith('MESH')]
        if len(xrefs) != 1:
            continue
        xref = xrefs[0]
        if mesh_client.get_mesh_name(xref, offline=True):
            mappings[node] = 'MESH:%s' % xref
    return mappings


# NOTE: these mappings are already integrated into equivalences.json
def get_cl_mesh_mappings():
    import re
    classdef_prefix = "# Class: obo:"
    mesh_id_pattern = re.compile(r'MESH:[CD][0-9]+')
    mappings = {}
    with open('/Users/ben/src/cell-ontology/src/ontology/cl-edit.owl', 'r') as fh:
        node = None
        for line in fh:
            if line.startswith(classdef_prefix):
                node_owl = line[len(classdef_prefix):len(classdef_prefix) + 10]
                node = node_owl.replace('_', ':')
            mesh_ids = set(mesh_id_pattern.findall(line))
            if node and len(mesh_ids) == 1:
                mappings[node] = list(mesh_ids)[0]
    return mappings


def get_display_name(ns: str) -> str:
    """Gets row/column name associated to a namespace"""
    return nmspace_displaynames[ns] if ns in nmspace_displaynames else ns


def f1(precision: float, recall: float) -> float:
    """Calculate the F1 score."""
    return 2 * precision * recall / (precision + recall)


@click.command()
@click.option(
    '--data',
    type=click.Path(dir_okay=True, file_okay=False),
    default=os.path.join(HERE, 'data'),
)
@click.option(
    '--results',
    type=click.Path(dir_okay=True, file_okay=False),
    default=os.path.join(HERE, 'results', "bioid_performance", __version__),
)
@click.option("--no-model-disambiguation", is_flag=True)
@click.option("--no-species-disambiguation", is_flag=True)
def main(data: str, results: str, no_model_disambiguation: bool, no_species_disambiguation: bool):
    """Run this script to evaluate gilda on the BioCreative VI BioID corpus.

    It has two optional arguments, --datapath and --resultspath that specify
    the path to the directory with necessary data and the path to the
    directory where results will be stored. The directory at datapath must
    contain a folder BIoIDtraining_2 containing the BIoID training corpus,
    and can optionally contain files equivalences.json
    serializing dictionaries of equivalence and isa relations between
    groundings. Results files will be added to the results directory in
    timestamped files.

    The data can be downloaded from
    https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz,
    and needs to be extracted in the benchmarks/data folder.
    """
    data_path = os.path.expandvars(os.path.expanduser(data))
    results_path = os.path.expandvars(os.path.expanduser(results))
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    try:
        with open(os.path.join(data_path, 'equivalences.json')) as f:
            equivalences = json.load(f)
    except FileNotFoundError:
        equivalences = {}
    benchmarker = BioIDBenchmarker(equivalences=equivalences)
    benchmarker.ground_entities_with_gilda(
        context=not no_model_disambiguation,
        species=not no_species_disambiguation,
    )
    print("Constructing mappings table...")
    mappings_table, mappings_table_unique = benchmarker.get_mappings_tables()
    print("Constructing results table...")
    counts, precision_recall, disamb_table = \
        benchmarker.get_results_tables(match='strict')
    print(
        f"Gilda v{__version__}, Bio-ontology v{bio_ontology.version},"
        f" model-based disambiguation={not no_model_disambiguation},"
        f" species-based disambiguation={not no_species_disambiguation}"
    )
    print(precision_recall.to_markdown(index=False))
    time = datetime.now().strftime('%y%m%d-%H%M%S')
    if no_model_disambiguation and no_species_disambiguation:
        outname = f'benchmark_no_disambiguation_{time}'
    elif no_model_disambiguation and not no_species_disambiguation:
        outname = f'benchmark_no_model_disambiguation_{time}'
    elif not no_model_disambiguation and no_species_disambiguation:
        outname = f'benchmark_no_species_disambiguation_{time}'
    else:
        outname = f'benchmark_{time}'

    # Generate output document
    caption0 = dedent(f"""\
    # Gilda Benchmarking

    Bio-ontology: v{bio_ontology.version}
    Gilda: v{__version__}
    Date: {time}
    """)
    caption1 = dedent("""\
        ## Table 1

        Mapping of groundings for entities in BioID corpus into namespaces used by
        Gilda. Count is by entries in corpus with groundings being counted multiple
        times if they occur in more than one entry. Some entries contain multiple
        equivalent curated groundings, leading to a discrepancy between the counts
        shown here and those in the other tables.
    """)
    table1 = mappings_table.to_markdown(index=False)
    caption2 = dedent("""\
        ## Table 2

        Mapping of groundings for entities in BioID corpus into Namespaces used by
        Gilda. Count is by unique groundings, with the same grounding only being
        counted once even if it appears in many entries.
    """)
    table2 = mappings_table_unique.to_markdown(index=False)
    caption3 = dedent("""\
        ## Table 3

        Counts of number of entries in corpus for each entity type, along with
        number of entries where Gilda's top grounding is correct, the number
        where one of Gilda's groundings is correct, and the number of entries
        where Gilda produced some grounding. Context based disambiguation is
        applied and Gilda's groundings are considered correct if there is
        an isa relation between the gold standard grounding and Gilda's or
        vice versa.
    """)
    table3 = counts.to_markdown(index=False)
    caption4 = dedent("""\
        ## Table 4

        Precision and recall values for Gilda performance by entity type. Values
        are given both for the case where Gilda is considered correct only if the
        top grounding matches and the case where Gilda is considered correct if
        any of its groundings match.
    """)
    table4 = precision_recall.to_markdown(index=False)
    caption5 = dedent("""\
        ## Table 5

        Comparison of results with and without context based disambiguation.
    """)
    table5 = disamb_table.to_markdown(index=False)
    output = '\n\n'.join([
        caption0,
        caption1, table1,
        caption2, table2,
        caption3, table3,
        caption4, table4,
        caption5, table5,
    ])
    result_stub = pathlib.Path(results_path).joinpath(outname)
    md_path = result_stub.with_suffix(".md")
    with open(md_path, 'w') as f:
        f.write(output)
    print(f'Output summary at {md_path}')

    latex_output = dedent(f'''\
    \\section{{Tables}}
        {mappings_table.to_latex(index=False, caption=caption1, label='tab:mappings')}
        {mappings_table_unique.to_latex(index=False, caption=caption2, label='tab:mappings-unique')}
        {counts.to_latex(index=False, caption=caption3, label='tab:counts')}
        {precision_recall.round(3).to_latex(index=False, caption=caption4, label='tab:precision-recall')}
        {disamb_table.to_latex(index=False, caption=caption5, label='tab:disambiguation')}
    ''')
    latex_path = result_stub.with_suffix(".tex")
    with open(latex_path, 'w') as file:
        print(latex_output, file=file)

    tsv_path = result_stub.with_suffix(".tsv")
    json_path = result_stub.with_suffix(".json")
    benchmarker.processed_data.to_csv(tsv_path, sep='\t', index=False)
    benchmarker.processed_data.to_json(json_path, force_ascii=False, orient="records", indent=2)


if __name__ == '__main__':
    main()
