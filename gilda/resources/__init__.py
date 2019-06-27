import os
import boto3
import logging

home_dir = os.path.expanduser('~')
resource_dir = os.path.join(home_dir, '.gilda', __version__)


if not os.path.isdir(resource_dir):
    try:
        os.makedirs(resource_dir)
    except Exception:
        logger.warning('%s already exists' % resource_dir)


def download_grounding_terms(path, base_name):
    config = botocore.client.Config(signature_version=botocore.UNSIGNED)
    s3 = boto3.client('s3', config=config)
    tc = boto3.s3.transfer.TransferConfig(use_threads=False)
    full_key = '%s/%s' % (__version__, base_name)
    s3.download_file('gilda', full_key, out_file, Config=tc)


def get_grounding_terms():
    base_name = 'grounding_terms.tsv'
    full_path = os.path.join(resource_dir, base_name))
    if not os.path.exists(full_path):
        download_grounding_terms(resource_dir, base_name)
    return full_path

