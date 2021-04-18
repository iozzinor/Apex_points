import argparse

from . import gradient
from .. import utils

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', nargs='?', default='.')
parser.add_argument('--debug', action='store_true')
parser.add_argument('json_file_paths', nargs='+')

arguments = parser.parse_args()

utils._localize_apices_from_json_file_paths(arguments.json_file_paths, gradient.localize_apices_engine, arguments.debug, arguments.output_dir, engine_name='gradient')

