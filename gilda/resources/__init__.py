import os
import boto3
import logging
import botocore
from gilda import __version__

logger = logging.getLogger(__name__)

HERE = os.path.abspath(os.path.dirname(__file__))
MESH_MAPPINGS_PATH = os.path.join(HERE, 'mesh_mappings.tsv')

home_dir = os.path.expanduser('~')
resource_dir = os.path.join(home_dir, '.gilda', __version__)


if not os.path.isdir(resource_dir):
    try:
        os.makedirs(resource_dir)
    except Exception:
        logger.warning('%s already exists' % resource_dir)

GROUNDING_TERMS_BASE_NAME = 'grounding_terms.tsv'
GROUNDING_TERMS_PATH = os.path.join(resource_dir, GROUNDING_TERMS_BASE_NAME)


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
    base_name = 'gilda_models.pkl'
    full_path = os.path.join(resource_dir, base_name)
    if not os.path.exists(full_path):
        logger.info('Downloading disambiguation models from S3.')
        out_file = _download_from_s3(resource_dir, base_name)
        logger.info('Saved disambiguation models into: %s' % out_file)
    return full_path
