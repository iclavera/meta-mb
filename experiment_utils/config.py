import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DOCKER_MOUNT_DIR = '/root/code/data'

DATA_DIR = os.path.join(BASE_DIR, 'data')

DOCKER_IMAGE = 'iclavera/meta-mb'

S3_BUCKET_NAME = 'iclavera-doodad'
