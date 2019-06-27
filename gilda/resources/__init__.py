import os
import boto3
import logging
import botocore
from gilda import __version__

logger = logging.getLogger(__name__)

home_dir = os.path.expanduser('~')
resource_dir = os.path.join(home_dir, '.gilda', __version__)


if not os.path.isdir(resource_dir):
    try:
        os.makedirs(resource_dir)
    except Exception:
        logger.warning('%s already exists' % resource_dir)


def download_grounding_terms(path, base_name):
    logger.info('Downloading grounding terms from S3.')
    config = botocore.client.Config(signature_version=botocore.UNSIGNED)
    s3 = boto3.client('s3', config=config)
    tc = boto3.s3.transfer.TransferConfig(use_threads=False)
    full_key = '%s/%s' % (__version__, base_name)
    out_file = os.path.join(path, base_name)
    s3.download_file('gilda', full_key, out_file, Config=tc)
    logger.info('Saved grounding terms into: %s' % out_file)


def get_grounding_terms():
    base_name = 'grounding_terms.tsv'
    full_path = os.path.join(resource_dir, base_name)
    if not os.path.exists(full_path):
        download_grounding_terms(resource_dir, base_name)
    return full_path

