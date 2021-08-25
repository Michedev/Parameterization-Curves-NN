import re
from argparse import ArgumentParser

import pandas as pd
from path import Path
import numpy as np
import yaml

runs = Path(__file__).parent / 'run'

def make_table_results(regex_result_file: str = r'test_result_([0-9]+)\.yaml', fname_output: str = 'result'):
    table = []
    for run in runs.dirs():
        for result_yaml in run.files('test_result_*.yaml'):
           m = re.match(regex_result_file, result_yaml.basename())
           if m:
               with open(result_yaml, 'r') as f:
                    data = yaml.load(f)
               iteration = m.group(1)
               iteration = int(iteration)
               data['iteration'] = iteration
               table.append(data)
    table = pd.DataFrame(table)
    print(table)
    print(table.loc[table.iteration == 10_000])
    table = table.loc[table.groupby('d')['avg_loss'].idxmin()]
    print(table.to_string())
    fname_output = f'{fname_output}.csv' if not fname_output.endswith('.csv') else fname_output
    table.to_csv(fname_output, index=False)
    return table

def setup_argparse():
    parser = ArgumentParser()
    parser.add_argument('-t', '--trigonometric', action='store_true', dest='trigonometric')
    return parser

if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    if args.trigonometric:
        make_table_results(r'test_result_trigonometric_([0-9]+)\.yaml', 'result_trigonometric')
    else:
        make_table_results()