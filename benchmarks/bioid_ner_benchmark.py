import os
import json
import pathlib
from collections import defaultdict

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
# MODULE = pystow.module('gilda', 'biocreative')
# URL = ('https://biocreative.bioinformatics.udel.edu/media/store/files/2017'
#        '/BioIDtraining_2.tar.gz')
DATA_DIR = os.path.join(HERE, 'data', 'BioIDtraining_2', 'caption_bioc')
ANNOTATIONS_PATH = os.path.join(HERE, 'data', 'BioIDtraining_2',
                                'annotations.csv')
RESULTS_DIR = os.path.join(HERE, 'results', "bioid_ner_performance",
                           gilda.__version__)
MODULE = pystow.module('gilda', 'biocreative')
URL = 'https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz'

tqdm.pandas()

BO_MISSING_XREFS = set()


class BioIDNERBenchmarker:
    def __init__(self):
        print("Instantiating benchmarker...")
        self.equivalences = self._load_equivalences()
        self.paper_level_grounding = defaultdict(set)
        self.processed_data = self.process_xml_files() #xml files processesed
        self.annotations_df = self._process_annotations_table() #csv annotations
        # self.reference_map = self.create_reference_map()  # Create reference map for efficient lookup
        self.gilda_annotations_map = defaultdict(list)  # New field to store Gilda annotations
        self.counts_table = None
        self.precision_recall = None

        # Print a small sample of annotations_df for debugging
        print("Sample of annotations_df:")
        print(self.annotations_df.head(10))  # Display first 10 rows
        print(self.annotations_df.columns)  # Display column names

        # Print unique values of doc_id and don_article for debugging
        # print("Unique doc_id values in processed_data:")
        # print(self.processed_data['id'].unique()[:10])  # Display first 10 unique IDs
        # print("Unique don_article values in annotations_df:")
        # print(self.annotations_df['don_article'].unique()[:10])  # Display first 10 unique IDs
        # Print unique values of doc_id and don_article for debugging
        print("First 10 unique doc_id values in processed_data:")
        print(self.processed_data['doc_id'].unique()[
              :10])  # Display first 10 unique IDs
        print("First 10 unique figure values in processed_data:")
        print(self.processed_data['figure'].unique()[
              :10])  # Display first 10 unique IDs
        print("First 10 unique don_article values in annotations_df:")
        print(self.annotations_df['don_article'].unique()[
              :10])  # Display first 10 unique IDs
        print("First 10 unique figure values in annotations_df:")
        print(self.annotations_df['figure'].unique()[
              :10])  # Display first 10 unique IDs

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
                    don_article, figure = doc_id_full.split(' ', 1)  # Split the full ID to get don_article and figure
                    don_article = don_article
                    for passage in document.findall('.//passage'):
                        offset = int(passage.find('.//offset').text)
                        text = passage.find('.//text').text
                        annotations = []
                        for annotation in passage.findall('.//annotation'):
                            annot_id = annotation.get('id')
                            annot_text = annotation.find('.//text').text
                            annot_type = annotation.find('.//infon[@key="type"]').text
                            annot_offset = int(annotation.find('.//location').attrib['offset'])
                            annot_length = int(annotation.find('.//location').attrib['length'])
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






        #         document = root.find('.//document')
        #         doc_id = document.find('.//id').text.strip()
        #         try:
        #             doc_id = int(doc_id)
        #         except ValueError:
        #             print(f"Skipping file with non-integer doc_id: {filename}")
        #             continue
        #
        #         text_elements = document.findall('.//text')
        #         texts = [elem.text for elem in text_elements if elem.text]
        #         full_text = ' '.join(texts)
        #
        #         if doc_id == 3868508:
        #             # Print the text being used for annotation for document ID 3868508
        #             print(f"Document ID: {doc_id}")
        #             print(
        #                 f"Full Text: {full_text[:500]}...")  # Print first 500 characters for brevity
        #
        #         data.append({'id': doc_id, 'text': full_text})
        # df = pd.DataFrame(data)
        # print(f"{len(df)} rows in processed XML data.")
        # return df

    def _load_equivalences(self) -> Dict[str, List[str]]:
        try:
            with open(os.path.join(DATA_DIR, 'equivalences.json')) as f:
                equivalences = json.load(f)
        except FileNotFoundError:
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
        # covered_fplx = {fplx_entry for fplx_entry, members
        #                 in fplx_members.items() if (members <= output)}
        # output |= {'FPLX:%s' % fplx_entry for fplx_entry in covered_fplx}
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
        df.loc[:, 'entity_type'] = df.apply(self._get_entity_type_helper, axis=1)
        processed_data = df[['text', 'obj', 'obj_synonyms', 'entity_type',
                             'don_article', 'figure', 'annot id', 'first left', 'last right']]
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

    def print_annotations_for_doc_id(self, doc_id):
        filtered_df = self.annotations_df[
            self.annotations_df['don_article'] == doc_id]
        print(f"Annotations for Document ID {doc_id}:")
        print(filtered_df)

    def annotate_entities_with_gilda(self):
        """Performs NER on the XML files using gilda.annotate()"""
        # df = self.processed_data
        tqdm.write("Annotating corpus with Gilda...")

        results = []
        # for _, row in df.iterrows():
        for _, item in self.processed_data.iterrows():
            doc_id = item['doc_id']
            figure = item['figure']
            text = item['text']
            # annotations = item['annotations']

            # Get the full text for the paper-level disambiguation
            full_text = self._get_plaintext(doc_id)

            gilda_annotations = annotate(text, context_text=full_text)

            for matched_text, scored_ner_match, start, end in gilda_annotations:
                grounding_results = ground(matched_text, context=full_text)

                for scored_grounding_match in grounding_results:

                    db, entity_id = grounding_results[0].term.db, grounding_results[0].term.id
                    curie = f"{db}:{entity_id}"
                    # normalized_id = self._normalize_id(curie)
                    synonyms = self.get_synonym_set([curie])
                    entity_type = self._get_entity_type([curie])

                    self.gilda_annotations_map[(doc_id, figure)].append({
                        'matched_text': matched_text,
                        'db': db,
                        'id': entity_id,
                        'start': start,
                        'end': end,
                        # 'normalized_id': normalized_id,
                        'synonyms': synonyms,
                        'entity_type': entity_type
                    })
                # results.append(
                #     (scored_match.term.db, scored_match.term.id, start, end))
                    if doc_id == '3868508' and figure == 'Figure_1-A':
                        print(f"Document ID: {doc_id}, Figure: {figure}")
                        print(f"Scored NER Match: {scored_ner_match}")
                        print(
                            f"Annotated Text Segment: {text[max(0, start - 50):min(len(text), end + 50)]}")
                        print(
                            f"Matched Text: {matched_text}, DB: {db}, ID: {entity_id}, Start: {start}, End: {end}")
                        print(f"Grounding Results: {curie}")
                        print(f"synonyms: {synonyms}")
                        print(f"entity type: {entity_type}")
            # annotations.append({'doc_id': doc_id, 'text': text, 'gilda_annotations': gilda_annotations})



        # df = pd.DataFrame(annotations)
        # self.processed_data = df
        # self.processed_data = pd.DataFrame(results)
        tqdm.write("Finished annotating corpus with Gilda...")

        # Print a small sample of processed_data for debugging
        # print("Sample of processed_data:")
        # print(self.processed_data.head(10))  # Display first 10 rows
        # print(self.processed_data.columns)  # Display column names

        # Update paper-level grounding with Gilda annotations
        # for _, row in df.iterrows():
        #     doc_id = row['doc_id']
        #     text = row['text']
        #     gilda_ann = row['gilda_annotations']
        #     for ann in gilda_ann:
        #         self.paper_level_grounding[(doc_id, text)].add(ann)

        # for doc_id, text, gilda_ann in zip(df['id'], df['text'],
        #                                    df['gilda_annotations']):
        #     for ann in gilda_ann:
        #         self.paper_level_grounding[(doc_id, text)].add(
        #             (ann[1].term.db, ann[1].term.id, ann[2], ann[3]))

    # def is_correct_annotation(self, row):
    #     """Cross-references gilda annotations with annotations
    #     provided by the dataset"""
    #     doc_id = int(row['doc_id'])
    #     figure = row['figure']
    #     db = row['db']
    #     annot_id = row['id']
    #     start = row['start']
    #     end = row['end']
    #
    #     # Ensure there are no leading/trailing spaces in the DataFrame
    #     self.annotations_df['figure'] = self.annotations_df['figure'].str.strip()
    #
    #     # ref_annotations = set()
    #
    #     ref_data = self.annotations_df[
    #         (self.annotations_df['don_article'] == doc_id) &
    #         (self.annotations_df['figure'] == figure)
    #         ]
    #
    #     # specific_doc_id = 3868508
    #     # specific_figure = 'Figure_1-A'
    #     # Debugging output to check why ref_data might be empty
    #     # if doc_id == specific_doc_id and figure == specific_figure:
    #     #     print(
    #     #         f"Checking annotations for Document ID: {doc_id}, Figure: {figure}")
    #     #     print(f"Annotations in DataFrame:\n{self.annotations_df.head()}")
    #     #     print(f"Filtered Reference Data:\n{ref_data}")
    #
    #     ref_annotations = set(
    #         (r['annot id'], r['first left'], r['last right'])
    #         for _, r in ref_data.iterrows()
    #     )
    #
    #     gilda_annotation = (annot_id, f'{db}:{row["id"]}')
    #
    #
    #     true_positive = gilda_annotation in ref_annotations
    #     false_positive = gilda_annotation not in ref_annotations
    #     false_negative = gilda_annotation not in ref_annotations
    #
    #     # expanded_gilda_annotations = {annot_id, db, row['start'], row['end']}
    #
    #     # for _, ref_row in ref_data.iterrows():
    #     #     ref_text = ref_row['obj']
    #     #     annot_id = ref_row['annot id']
    #     #     start_pos = int(ref_row['first left'])
    #     #     end_pos = int(ref_row['last right'])
    #     #     for normalized_id in ref_text:
    #     #         parts = normalized_id.split(':')
    #     #         db, identifier = parts[0], ':'.join(parts[1:])
    #     #         ref_annotations.add((annot_id, db, identifier, start_pos, end_pos))
    #     #
    #           # Use synonym sets to match equivalent terms
    #     # expanded_gilda_annotations = set()
    #     # for ann in gilda_annotations:
    #     #     expanded_gilda_annotations.update(
    #     #         self.get_synonym_set([f"{ann[0]}:{ann[1]}"]))
    #
    #     # Conditional print for debugging: Limit to a small set of data
    #     # if doc_id == specific_doc_id and figure == specific_figure:
    #     #     print("Gilda Annotations:")
    #     #     print(list(gilda_annotations)[:5])  # Print first few annotations for brevity
    #     #     print("Reference Annotations:")
    #     #     print(list(ref_annotations)[:5]) # Print first few annotations for brevity
    #
    #     # return expanded_gilda_annotations, ref_annotations
    #
    #     # true_positives = 0
    #     # false_positives = 0
    #     # false_negatives = 0
    #
    #     # Calculate true positives and false positives
    #     # for gilda_ann in expanded_gilda_annotations:
    #     #     if gilda_ann in ref_annotations:
    #     #         true_positives += 1
    #     #     else:
    #     #         false_positives += 1
    #     #
    #     # # Calculate false negatives
    #     # for ref_ann in ref_annotations:
    #     #     if ref_ann not in expanded_gilda_annotations:
    #     #         false_negatives += 1
    #
    #     return true_positive, false_positive, false_negative

    # def check_annotation(self, row, reference_map):
    #     """Checks if the Gilda annotation exists in the reference annotations"""
    #     doc_id = int(row['doc_id'])
    #     figure = row['figure']
    #     gilda_annotation = f'{row["db"]}:{row["id"]}'
    #     gilda_synonyms = self.get_synonym_set([gilda_annotation])
    #
    #     if (doc_id, figure) in reference_map:
    #         ref_annotations = reference_map[(doc_id, figure)]
    #
    #         true_positive = any(
    #             syn in ref_annotations for syn in gilda_synonyms)
    #         false_positive = not true_positive
    #         false_negative = not true_positive
    #
    #         return int(true_positive), int(false_positive), int(false_negative)
    #
    #     return 0, 1, 0  # If there are no reference annotations, it's a false positive

    def evaluate_gilda_performance(self):
        """Calculates precision, recall, and F1"""
        print("Evaluating performance...")
        # df = self.processed_data

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        # reference_map = self.create_reference_map()

        for (doc_id, figure), annotations in self.gilda_annotations_map.items():
            # print(f"Processing Document ID: {doc_id}, Figure: {figure}")
            for annotation in annotations:
                start = annotation['start']
                end = annotation['end']
                # gilda_annotation = annotation['id']
                gilda_synonyms = annotation['synonyms']
                text = annotation['matched_text']

                # gilda_annotation = f'{annotation["db"]}:{annotation["id"]}'
                # gilda_synonyms = self.get_synonym_set([gilda_annotation])

                # ref_annotations = self.reference_map.get((doc_id, figure), set())

                ref_annotations = self.annotations_df[
                    (self.annotations_df['don_article'] == int(doc_id)) &
                    (self.annotations_df['figure'] == figure)
                    ]



                # Check if any synonym of the Gilda annotation matches reference annotations
                # match_found = any((text, syn, start, end) in ref_annotations for syn in gilda_synonyms)

                match_found = any(
                    (text, syn, start, end) in ref_annotations[
                        ['text', 'obj_synonyms', 'first left',
                         'last right']].values
                    for syn in gilda_synonyms
                )


                if doc_id == '3868508' and figure == "Figure_1-A":
                    print(f"Document ID: {doc_id}, Figure: {figure}")
                    print(f"Gilda Annotation: {annotation}")
                    print(f"Reference Annotations: {ref_annotations}")
                    print(f"Match Found: {match_found}")

                if match_found:
                    total_true_positives += 1
                else:
                    total_false_positives += 1
                    total_false_negatives += 1

                # total_true_positives += int(true_positive)
                # total_false_positives += int(false_positive)
                # total_false_negatives += int(false_negative)

            # tp, fp, fn = self.check_annotation(row, reference_map)

                # = self.is_correct_annotation(row)
            # total_true_positives += tp
            # total_false_positives += fp
            # total_false_negatives += fn



        # for data in self.process_xml_files():
        #     doc_id = data['doc_id']
        #     text = data['text']
        #     gilda_ann = data['gilda_annotations']
        #     figure = data.get('figure','')  # Ensure figure is retrieved if it exists

        precision = total_true_positives / (
                    total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        recall = total_true_positives / (
                    total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # df = self.processed_data
        # results = df.apply(self.is_correct_annotation, axis=1)

        # df['true_positives'] = results.apply(lambda x: x[0])
        # df['false_positives'] = results.apply(lambda x: x[1])
        # df['false_negatives'] = results.apply(lambda x: x[2])

        # total_true_positives = df['true_positives'].sum()
        # total_false_positives = df['false_positives'].sum()
        # total_false_negatives = df['false_negatives'].sum()

        # precision = total_true_positives / (total_true_positives
        #                                     + total_false_positives) \
        #     if (total_true_positives + total_false_positives) > 0 else 0
        #
        # recall = total_true_positives / (total_true_positives +
        #                                  total_false_negatives) \
        #     if (total_true_positives + total_false_negatives) > 0 else 0
        #
        # f1 = 2 * (precision * recall) / (precision + recall) \
        #     if (precision + recall) > 0 else 0

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

        print("Aggregated Results:")
        print(counts_table)
        print(precision_recall)

        self.counts_table = counts_table
        self.precision_recall = precision_recall

        print("Finished evaluating performance...")

    # def create_reference_map(self):
    #     """Creates a hashmap (dictionary) of reference annotations for efficient lookup"""
    #     reference_map = defaultdict(set)
    #     print("Creating reference map...")
    #
    #     for _, row in self.annotations_df.iterrows():
    #         doc_id = str(row['don_article'])
    #         figure = row['figure']
    #         start = row['first left']
    #         end = row['last right']
    #         obj = row['obj']  # Use the object directly
    #         text = row['text']
    #
    #         for original_id in obj:
    #             reference_map[(doc_id, figure)].add(
    #                 (text, original_id, start, end))
    #
    #         for synonym in row['obj_synonyms']:
    #             reference_map[(doc_id, figure)].add((text, synonym, start, end,))
    #
    #         # ref_obj_synonyms = self.get_synonym_set(row['obj'])
    #         # for syn in ref_obj_synonyms:
    #         #     reference_map[(doc_id, figure)].add((syn, start, end))
    #
    #         if doc_id == '3868508' and figure == "Figure_1-A":
    #             print(
    #                 f"Adding Reference Annotation: {(text, obj, start, end)} for Document ID: {doc_id}, Figure: {figure}")
    #
    #         # reference_map[(doc_id, figure)].append({
    #         #     'text': row['text'],
    #         #     'obj': row['obj'],
    #         #     'obj_synonyms': row['obj_synonyms'],
    #         #     'entity_type': row['entity_type'],
    #         #     'annot_id': row['annot id'],
    #         #     'first_left': row['first left'],
    #         #     'last_right': row['last right']
    #         # })
    #
    #     return reference_map

    def get_results_tables(self):
        return self.counts_table, self.precision_recall


# def get_famplex_members():
#     from indra.databases import hgnc_client
#     entities_path = os.path.join(HERE, 'data', 'entities.csv')
#     fplx_entities = famplex.load_entities()
#     fplx_children = defaultdict(set)
#     for fplx_entity in fplx_entities:
#         members = famplex.individual_members('FPLX', fplx_entity)
#         for db_ns, db_id in members:
#             if db_ns == 'HGNC':
#                 db_id = hgnc_client.get_current_hgnc_id(db_id)
#                 if db_id:
#                     fplx_children[fplx_entity].add('%s:%s' % (db_ns, db_id))
#     return dict(fplx_children)
#
#
# fplx_members = get_famplex_members()


# def main(results: str):
def main():
    # results_path = os.path.expandvars(os.path.expanduser(results))
    # os.makedirs(results_path, exist_ok=True)

    benchmarker = BioIDNERBenchmarker()
    benchmarker.print_annotations_for_doc_id(3868508)
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
