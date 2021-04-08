import argparse

import apex_points

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', nargs='?', default='.')
parser.add_argument('--debug', action='store_true')
parser.add_argument('json_file_paths', nargs='+')

arguments = parser.parse_args()

apex_points.localize_apices(arguments.json_file_paths, arguments.debug, arguments.output_dir)

