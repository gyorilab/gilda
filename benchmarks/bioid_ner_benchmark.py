import os
import json
import pathlib
from collections import defaultdict, Counter

from functools import lru_cache
import pandas as pd
import xml.etree.ElementTree as ET

from lxml import etree
from tqdm import tqdm
from datetime import datetime
from typing import List, Tuple, Set, Dict, Optional, Iterable, Collection
import click
import pystow
import gilda
from gilda import ground
# from benchmarks.bioid_evaluation import fplx_members
from gilda.ner import annotate
from gilda.grounder import logger

import famplex
from indra.databases.chebi_client import get_chebi_id_from_pubchem
from indra.databases.hgnc_client import get_hgnc_from_entrez
from indra.databases.uniprot_client import get_hgnc_id
from indra.ontology.bio import bio_ontology

logger.setLevel('WARNING')

# Constants
HERE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(HERE, 'data', 'BioIDtraining_2', 'caption_bioc')
ANNOTATIONS_PATH = os.path.join(HERE, 'data', 'BioIDtraining_2',
                                'annotations.csv')
RESULTS_DIR = os.path.join(HERE, 'results', "bioid_ner_performance",
                           gilda.__version__)
MODULE = pystow.module('gilda', 'biocreative')
URL = 'https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz'

STOPLIST_PATH = os.path.join(HERE, 'data', 'ner_stoplist.txt')

tqdm.pandas()

BO_MISSING_XREFS = set()


class BioIDNERBenchmarker:
    def __init__(self):
        print("Instantiating benchmarker...")
        self.equivalences = self._load_equivalences()
        self.paper_level_grounding = defaultdict(set)
        self.processed_data = self.process_xml_files()  # xml files processesed
        self.annotations_df = self._process_annotations_table()  # csv annotations
        # self.reference_map = self.create_reference_map()  # Create reference map for efficient lookup
        self.stoplist = self._load_stoplist()  # Load stoplist
        self.gilda_annotations_map = defaultdict(
            list)  # New field to store Gilda annotations
        self.counts_table = None
        self.precision_recall = None

    def process_xml_files(self):
        """Extract relevant information from XML files."""
        print("Extracting information from XML files...")
        data = []
        total_annotations = 0
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.xml'):
                filepath = os.path.join(DATA_DIR, filename)
                tree = ET.parse(filepath)
                root = tree.getroot()
                for document in root.findall('.//document'):
                    doc_id_full = document.find('.//id').text.strip()
                    don_article, figure = doc_id_full.split(' ',
                                                            1)  # Split the full ID to get don_article and figure
                    don_article = don_article
                    for passage in document.findall('.//passage'):
                        offset = int(passage.find('.//offset').text)
                        text = passage.find('.//text').text
                        annotations = []
                        for annotation in passage.findall('.//annotation'):
                            annot_id = annotation.get('id')
                            annot_text = annotation.find('.//text').text
                            annot_type = annotation.find(
                                './/infon[@key="type"]').text
                            annot_offset = int(
                                annotation.find('.//location').attrib['offset'])
                            annot_length = int(
                                annotation.find('.//location').attrib['length'])
                            annotations.append({
                                'annot_id': annot_id,
                                'annot_text': annot_text,
                                'annot_type': annot_type,
                                'annot_offset': annot_offset,
                                'annot_length': annot_length,
                            })
                            total_annotations += 1
                        data.append({
                            'doc_id': don_article,
                            'figure': figure,
                            'offset': offset,
                            'text': text,
                            'annotations': annotations,
                        })
        # df = pd.DataFrame(data)
        # print(f"{len(df)} rows in processed XML data.")
        print(f"Total annotations in XML files: {total_annotations}")
        print("Finished extracting information from XML files.")
        return pd.DataFrame(data)

    def _load_stoplist(self) -> Set[str]:
        """Load NER stoplist from file."""
        stoplist_path = STOPLIST_PATH
        try:
            with open(stoplist_path, 'r') as file:
                stoplist = {line.strip().lower() for line in file}
            print(f"Loaded stoplist with {len(stoplist)} entries.")
            return stoplist
        except FileNotFoundError:
            print(
                f"No stoplist found at {stoplist_path}. Proceeding without it.")
            return set()

    def _load_equivalences(self) -> Dict[str, List[str]]:
        try:
            with open(os.path.join(HERE, 'data', 'equivalences.json')) as f:
                equivalences = json.load(f)
        except FileNotFoundError:
            print(
                f"No Equivalences found at {os.path.join(HERE, 'data', 'equivalences.json')}. Proceeding without it.")
            equivalences = {}
        return equivalences

    @classmethod
    def _normalize_ids(cls, curies: str) -> List[Tuple[str, str]]:
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
        for xref_prefix, xref_id in bio_ontology.get_mappings(prefix,
                                                              identifier):
            output.add(f'{xref_prefix}:{xref_id}')

        # TODO these should all be in bioontology, eventually
        for xref_curie in self.equivalences.get(curie, []):
            if xref_curie in output:
                continue
            xref_prefix, xref_id = xref_curie.split(':', maxsplit=1)
            if (prefix, xref_prefix) not in BO_MISSING_XREFS:
                BO_MISSING_XREFS.add((prefix, xref_prefix))
                tqdm.write(
                    f'Bioontology v{bio_ontology.version} is missing mappings from {prefix} to {xref_prefix}')
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

    def _get_entity_type_helper(self, row) -> str:
        if self._get_entity_type(row.obj) != 'Gene':
            return self._get_entity_type(row.obj)
        elif any(y.startswith('HGNC') for y in row.obj_synonyms):
            return 'Human Gene'
        else:
            return 'Nonhuman Gene'

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
        elif any(
                grounding.startswith('CHEBI') or grounding.startswith('PubChem')
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
        elif any(
                grounding.startswith('NCBI taxon') for grounding in groundings):
            return 'Taxon'
        else:
            return 'unknown'

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
        df.loc[:, 'entity_type'] = df.apply(self._get_entity_type_helper,
                                            axis=1)
        processed_data = df[['text', 'obj', 'obj_synonyms', 'entity_type',
                             'don_article', 'figure', 'annot id', 'first left',
                             'last right']]
        print("%d rows in processed annotations table." % len(processed_data))
        processed_data = processed_data[processed_data.entity_type
                                        != 'unknown']
        print("%d rows in annotations table without unknowns." %
              len(processed_data))
        for don_article, text, synonyms in df[['don_article', 'text',
                                               'obj_synonyms']].values:
            self.paper_level_grounding[don_article, text].update(synonyms)
        return processed_data

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

    def annotate_entities_with_gilda(self):
        """Performs NER on the XML files using gilda.annotate()"""
        # df = self.processed_data
        tqdm.write("Annotating corpus with Gilda...")

        # results = []
        total_gilda_annotations = 0
        # for _, row in df.iterrows():
        for _, item in self.processed_data.iterrows():
            doc_id = item['doc_id']
            figure = item['figure']
            text = item['text']
            # annotations = item['annotations']

            # Get the full text for the paper-level disambiguation
            full_text = self._get_plaintext(doc_id)

            gilda_annotations = annotate(text, context_text=full_text,
                                         return_first=True)
            # for testing all matches for each entity, return_first = False.

            for matched_text, grounding_result, start, end in gilda_annotations:

                # Checking against stoplist
                if matched_text in self.stoplist:
                    continue

                db, entity_id = grounding_result.term.db, grounding_result.term.id
                curie = f"{db}:{entity_id}"
                # normalized_id = self._normalize_id(curie)
                synonyms = self.get_synonym_set([curie])
                # entity_type = self._get_entity_type([curie])

                self.gilda_annotations_map[(doc_id, figure)].append({
                    'matched_text': matched_text,
                    'db': db,
                    'id': entity_id,
                    'start': start,
                    'end': end,
                    # 'normalized_id': normalized_id,
                    'synonyms': synonyms,
                    # 'entity_type': entity_type
                })
                total_gilda_annotations += 1
                # results.append(
                #     (scored_match.term.db, scored_match.term.id, start, end))
                if doc_id == '3868508' and figure == 'Figure_1-A':
                    print(f"Scored NER Match: {grounding_result}")
                    print(f"Annotated Text Segment: {text[start:end]} at "
                          f"indices {start} to {end}")
                    print(
                        f"Gilda Matched Text: {matched_text}, DB: {db}, "
                        f"ID: {entity_id}, Start: {start}, End: {end}")
                    print(f"Grounding Results: {curie}")
                    print(f"synonyms: {synonyms}")
                    # print(f"entity type: {entity_type}")
                    print("\n")

        tqdm.write("Finished annotating corpus with Gilda...")
        print(f"Total Gilda annotations: {total_gilda_annotations}")

    def evaluate_gilda_performance(self):
        """Calculates precision, recall, and F1"""
        print("Evaluating performance...")

        # df = self.processed_data

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        false_positives_counter = Counter()

        # Create a set of reference annotations for quick lookup
        ref_annotations = set()
        for _, row in self.annotations_df.iterrows():
            doc_id = str(row['don_article'])
            figure = row['figure']
            text = row['text']
            for syn in row['obj_synonyms']:
                ref_annotations.add((doc_id, figure, text, syn,
                                           row['first left'],
                                           row['last right']))

        print(f"Total reference annotations: {len(ref_annotations)}")
        for (doc_id, figure), annotations in self.gilda_annotations_map.items():
            # print(f"Processing Document ID: {doc_id}, Figure: {figure}")
            for annotation in annotations:
                start = annotation['start']
                end = annotation['end']
                # gilda_annotation = annotation['id']
                gilda_synonyms = annotation['synonyms']
                text = annotation['matched_text']

                # Uncomment this if above doesnt work.
                # ref_annotations = self.annotations_df[
                #     (self.annotations_df['don_article'] == int(doc_id)) &
                #     (self.annotations_df['figure'] == figure)
                #     ]

                # UNCOMMENT IF BELOW DOESN'T WORK
                # match_found = any(
                #     (text, syn, start, end) in ref_annotations[
                #         ['text', 'obj_synonyms', 'first left',
                #          'last right']].values
                #     for syn in gilda_synonyms
                # )

                match_found = any(
                    (doc_id, figure, text, syn, start, end)
                    in ref_annotations for syn in gilda_synonyms)

                # Debugging: Identify and print the exact match
                matching_reference = None
                if match_found:
                    for syn in gilda_synonyms:
                        if (doc_id, figure, text, syn, start,
                            end) in ref_annotations:
                            matching_reference = (
                            doc_id, figure, text, syn, start, end)
                            break

                if (match_found == True and doc_id == '3868508'
                        and figure == "Figure_1-A"):
                    print(f"Gilda Annotation: {annotation}")
                    # print(f"Reference Annotations: {ref_annotations}")
                    print(f"Match Found: {match_found}")
                    print(f"Synonyms: {gilda_synonyms}")
                    print(f"Match Found: {match_found}")
                    if match_found:
                        print(f"Matching Reference: {matching_reference}")

                if match_found:
                    total_true_positives += 1
                else:
                    total_false_positives += 1
                    false_positives_counter[text] += 1
                    # total_false_negatives += 1
        print(f"20 Most Common False Positives: "
              f"{false_positives_counter.most_common(20)}")

        for doc_id, figure, text, syn, start, end in ref_annotations:
            gilda_annotations = self.gilda_annotations_map.get((doc_id, figure),
                                                               [])
            match_found = any(
                ann['matched_text'] == text and
                syn in ann['synonyms'] and
                ann['start'] == start and
                ann['end'] == end
                for ann in gilda_annotations
            )
            if not match_found:
                total_false_negatives += 1




        # UNCOMMENT IF ABOVE DOESNT WORK
        # for _, row in self.annotations_df.iterrows():
        #     doc_id, figure = row['don_article'], row['figure']
        #     new_ref_annotation = (
        #         row['text'], set(row['obj_synonyms']), row['first left'],
        #         row['last right'])
        #
        #     # Check if this reference annotation is in Gilda's annotations
        #     new_gilda_annotations = self.gilda_annotations_map.get(
        #         (str(doc_id), figure), [])
        #     ref_match_found = False
        #     for ann in new_gilda_annotations:
        #         # (new_ref_annotation[1].intersection(ann['synonyms']) and
        #         if new_ref_annotation[0] == ann['matched_text'] and \
        #                 new_ref_annotation[2] == ann['start'] and \
        #                 new_ref_annotation[3] == ann['end']:
        #             ref_match_found = True
        #             break
        #
        #     if not ref_match_found:
        #         total_false_negatives += 1

        precision = total_true_positives / (total_true_positives
                                            + total_false_positives) \
            if (total_true_positives + total_false_positives) > 0 else 0.0

        recall = total_true_positives / (total_true_positives
                                          + total_false_negatives) \
            if (total_true_positives + total_false_negatives) > 0 else 0.0

        f1 = (2 * (precision * recall)) / (precision + recall) \
            if ((precision + recall) > 0) else 0

        counts_table = pd.DataFrame([{
            'True Positives': total_true_positives,
            'False Positives': total_false_positives,
            'False Negatives': total_false_negatives
        }])

        precision_recall = pd.DataFrame([{
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }])

        self.counts_table = counts_table
        self.precision_recall = precision_recall

        os.makedirs(RESULTS_DIR, exist_ok=True)
        false_positives_df = pd.DataFrame(false_positives_counter.items(),
                                          columns=['False Positive Text',
                                                   'Count'])
        false_positives_df = false_positives_df.sort_values(by='Count', ascending=False)
        false_positives_df.to_csv(
            os.path.join(RESULTS_DIR, 'false_positives.csv'), index=False)

        print("Finished evaluating performance...")

    def get_results_tables(self):
        return self.counts_table, self.precision_recall


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


# def main(results: str):
def main():
    # results_path = os.path.expandvars(os.path.expanduser(results))
    # os.makedirs(results_path, exist_ok=True)

    benchmarker = BioIDNERBenchmarker()
    benchmarker.annotate_entities_with_gilda()
    benchmarker.evaluate_gilda_performance()
    counts, precision_recall = benchmarker.get_results_tables()
    print("Counts table:")
    print(counts.to_markdown(index=False))
    print("Precision and Recall table:")
    print(precision_recall.to_markdown(index=False))
    # time = datetime.now().strftime('%y%m%d-%H%M%S')
    # result_stub = pathlib.Path(results_path).joinpath(f'benchmark_{time}')
    # counts.to_csv(result_stub.with_suffix(".counts.csv"), index=False)
    # precision_recall.to_csv(result_stub.with_suffix(".precision_recall.csv"),
    #                         index=False)
    # print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()
