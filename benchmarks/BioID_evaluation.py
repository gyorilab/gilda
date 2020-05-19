import pandas as pd

import os
import lxml.etree as etree

from indra.databases.uniprot_client import get_hgnc_id
from indra.databases.hgnc_client import get_hgnc_from_entrez
from indra.databases.chebi_client import get_chebi_id_from_pubchem

from gilda.grounder import logger, Grounder

logger.setLevel('WARNING')


class GroundingEvaluator(object):
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
    """
    def __init__(self, bioid_data_path, grounder=None,
                 equivalences=None, isa_relations=None):
        if grounder is None:
            grounder = Grounder()
        if equivalences is None:
            equivalences = {}
        available_namespaces = set()
        for terms in grounder.entries.values():
            for term in terms:
                available_namespaces.add(term.db)
        self.grounder = grounder
        self.equivalences = equivalences
        self.available_namespaces = available_namespaces
        self.bioid_data_path = bioid_data_path
        self.processed_data = self._process_annotations_table()

    def _process_annotations_table(self):
        """Extract relevant information from annotations table."""
        df = pd.read_csv(os.path.join(self.bioid_data_path,
                                      'annotations.csv'),
                         sep=',', low_memory=False)
        # Split entries with multiple groundings
        df.loc[:, 'obj'] = df['obj'].\
            apply(lambda x: x.split('|'))
        # Create column for entity type
        df['entity_type'] = df['obj'].\
            apply(lambda x: self._get_entity_type(x))
        # Normalize ids
        df.loc[:, 'obj'] = df['obj'].\
            apply(lambda x: [self._normalize_id(y) for y in x])
        # Add synonyms of gold standard groundings to help match more things
        df['obj_synonyms'] = df['obj'].\
            apply(lambda x: self.get_synonym_set(x))
        # Find gilda groundings for entity text with and without context
        df['gilda_groundings_no_context'] = df.text.\
            apply(lambda x: self._get_grounding_list(x))
        df['gilda_groundings'] = df.\
            apply(lambda row:
                  self._get_grounding_list(
                      row.text,
                      context=self._get_plaintext(row.don_article)), axis=1)
        processed_data = df[['text', 'obj', 'obj_synonyms', 'entity_type',
                             'don_article', 'gilda_groundings_no_context',
                             'gilda_groundings']]
        return processed_data

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
        if any([x.startswith('NCBI gene')
                or x.startswith('Uniprot') for x in bioc_groundings]):
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
        elif any([x.startswith('Uberon') for x in bioc_groundings]):
            result = 'Tissue/Organ'
        elif any([x.startswith('NCBI taxon') for x in bioc_groundings]):
            result = 'Taxon'
        else:
            result = 'unknown'
        return result

    def _get_grounding_list(self, text, context=None):
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
        output = set(id_)
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
