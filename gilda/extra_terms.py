import itertools as itt
import json
import logging
import os

import indra
from indra.databases import doid_client, mesh_client
from tqdm import tqdm

from gilda.process import normalize
from gilda.term import Term

indra_module_path = indra.__path__[0]
resources = os.path.join(indra_module_path, 'resources')

logger = logging.getLogger(__name__)


def generate_doid_terms():
    return _generate_obo_terms('doid')


def generate_efo_terms():
    return _generate_obo_terms('efo')


def generate_hp_terms():
    return _generate_obo_terms('hp')


def _generate_obo_terms(prefix):
    filename = os.path.join(resources, '{prefix}.json'.format(prefix=prefix))
    logger.info('Loading %s', filename)
    with open(filename) as file:
        entries = json.load(file)

    terms = []
    for entry in tqdm(entries, desc=prefix):
        db, db_id, db_name = prefix.upper(), entry['id'], entry['name']
        name_term = Term(
            norm_text=normalize(db_name),
            text=db_name,
            db=db,
            id=db_id,
            entry_name=db_name,
            status='name',
            source=prefix,
        )
        terms.append(name_term)

        entities = [
            (db, db_id, db_name),
        ]
        # TODO add more entities based on xrefs?
        for xref in entry['xrefs']:
            xref_db, xref_db_id = xref['namespace'], xref['id']
            if xref_db_id == 'NoID':
                continue
            if xref_db in {'MESH', 'MSH'}:
                mesh_name = mesh_client.get_mesh_name(xref_db_id, offline=True)
                if mesh_name is not None:
                    entities.append(('MESH', xref_db_id, mesh_name))
                else:
                    logger.info('Could not find MESH xref %s', xref_db_id)
            elif xref_db == 'DOID':
                if not xref_db_id.startswith('DOID:'):
                    xref_db_id = 'DOID:' + xref_db_id
                doid_name = doid_client.get_doid_name_from_doid_id(xref_db_id)
                if doid_name is None:
                    doid_canonical_id = doid_client.get_doid_id_from_doid_alt_id(xref_db_id)
                    if doid_canonical_id is not None:
                        doid_name = doid_client.get_doid_name_from_doid_id(doid_canonical_id)
                if doid_name is not None:
                    entities.append(('DOID', xref, doid_name))
                else:
                    logger.info('Could not find DOID xref %s', xref_db_id)

        synonyms = set(entry['synonyms'])
        for synonym, (db, db_id, db_name) in itt.product(synonyms, entities):
            synonym_term = Term(
                norm_text=normalize(synonym),
                text=synonym,
                db=db,
                id=db_id,
                entry_name=db_name,
                status='synonym',
                source=prefix,
            )
            terms.append(synonym_term)

    logger.info('Loaded %d terms from %s', len(terms), prefix)
    return terms
