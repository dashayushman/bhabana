import tarfile
import argparse


parser = argparse.ArgumentParser(
        description='This script receives names of models or datasets and '
                    'downloads them for being used by bhabana')
# Add arguments
parser.add_argument('-n', '--name', type=str,
                  help='Name of the model or dataset to download',
                  required=True)
ARGS = parser.parse_args()



def maybe_download(name):
    if not is_supported(name):
        raise ValueError('Dataset or Model with name {} is not supported in '
                         'any of the release versions.')
    pass


if __name__ == '__main__':
    maybe_download(ARGS.name)
