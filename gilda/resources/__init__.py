import os
import boto3
import pystow
import logging
import botocore
from gilda import __version__

logger = logging.getLogger(__name__)

HERE = os.path.abspath(os.path.dirname(__file__))
MESH_MAPPINGS_PATH = os.path.join(HERE, 'mesh_mappings.tsv')

resource_dir = pystow.join('gilda', __version__)

GROUNDING_TERMS_BASE_NAME = 'grounding_terms.tsv.gz'
GROUNDING_TERMS_PATH = os.path.join(resource_dir, GROUNDING_TERMS_BASE_NAME)


# Popular organisms per UniProt, see
# https://www.uniprot.org/help/filter_options
popular_organisms = ['9606', '10090', '10116', '9913', '7955', '7227',
                     '6239', '44689', '3702', '39947', '83333', '224308',
                     '559292']

organism_labels = {
    '9606': 'Homo sapiens',
    '10090': 'Mus musculus',
    '10116': 'Rattus norvegicus',
    '9913': 'Bos taurus',
    '7955': 'Danio rerio',
    '7227': 'Drosophila melanogaster',
    '6239': 'Caenorhabditis elegans',
    '44689': 'Dictyostelium discoideum',
    '3702': 'Arabidopsis thaliana',
    '39947': 'Oryza sativa',
    '83333': 'Escherichia coli',
    '224308': 'Bacillus subtilis',
    '559292': 'Saccharomyces cerevisiae',
}

# NOTE: these are not all exact mappings..
# Several mappings here are to the closest match which works correctly
# in this setting but isn't generally speaking a valid xref.
taxonomy_to_mesh = {
    '9606': 'D006801',
    '10090': 'D051379',
    '10116': 'D051381',
    '9913': 'D002417',
    '7955': 'D015027',
    '7227': 'D004331',
    '6239': 'D017173',
    '44689': 'D004023',
    '3702': 'D017360',
    '39947': 'D012275',
    '83333': 'D048168',
    '224308': 'D001412',
    '559292': 'D012441',
}
mesh_to_taxonomy = {v: k for k, v in taxonomy_to_mesh.items()}


def _download_from_s3(path, base_name):
    config = botocore.client.Config(signature_version=botocore.UNSIGNED)
    s3 = boto3.client('s3', config=config)
    tc = boto3.s3.transfer.TransferConfig(use_threads=False)
    full_key = '%s/%s' % (__version__, base_name)
    out_file = os.path.join(path, base_name)
    s3.download_file('gilda', full_key, out_file, Config=tc)
    return out_file


def get_grounding_terms():
    base_name = GROUNDING_TERMS_BASE_NAME
    full_path = GROUNDING_TERMS_PATH
    if not os.path.exists(full_path):
        logger.info('Downloading grounding terms from S3.')
        out_file = _download_from_s3(resource_dir, base_name)
        logger.info('Saved grounding terms into: %s' % out_file)
    return full_path


def get_gilda_models():
    base_name = 'gilda_models.json.gz'
    full_path = os.path.join(resource_dir, base_name)
    if not os.path.exists(full_path):
        logger.info('Downloading disambiguation models from S3.')
        out_file = _download_from_s3(resource_dir, base_name)
        logger.info('Saved disambiguation models into: %s' % out_file)
    return full_path
