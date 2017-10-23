import argparse
import bhabana.utils as utils
import logging.config

from bhabana.utils import data_utils as du


logger = logging.getLogger(__name__)
logging.basicConfig()

parser = argparse.ArgumentParser(
        description='This script receives names of models or datasets and '
                    'downloads them for being used by bhabana')
# Add arguments
parser.add_argument('-n', '--name', type=str,
                  help='Name of the model or dataset to download. E.g., '
                       '"amazon_reviews_de" or "hotel_reviews_en"',
                  required=True)

parser.add_argument('-t', '--type', type=str,
                  help='Type of dat that you want to download. E.g., "dataset" '
                       'or "model"',
                  required=True)

ARGS = parser.parse_args()

if __name__ == '__main__':
    logger.info('maja')
    du.maybe_download(ARGS.name, ARGS.type, force=True)
