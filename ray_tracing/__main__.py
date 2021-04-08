import argparse

from . import ray_tracing

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', nargs='?', default='.')
parser.add_argument('--debug', action='store_true')
parser.add_argument('json_file_paths', nargs='+')

arguments = parser.parse_args()

ray_tracing.localize_apices_from_json_file_paths(arguments.json_file_paths, arguments.debug, arguments.output_dir)

