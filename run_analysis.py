import re
from argparse import ArgumentParser
from typing import Optional, List

import pandas as pd
from path import Path
import numpy as np
import yaml


ROOT: Path = Path(__file__).parent.abspath()
RUNS_DIR: Path = ROOT / 'run'

REGEX_TRIGONOMETRIC_RESULT_FILE = r'test_result_trigonometric_(?P<iteration>[0-9]+)_([0-9]+)\.yaml'
REGEX_RESULT_FILE = r'test_result_(?P<iteration>[0-9]+)\.yaml'


def avg_tables(tables):
    result = tables[0].copy(deep=True)
    for i in range(1, len(tables)):
        result = result + tables[i]
    result = result / len(tables)
    return result


def gather_results_run(run_folder, regex_result_file: str, gather_parameters) -> pd.DataFrame:
    table = []
    for result_yaml in run_folder.files('test_result_*.yaml'):
        m = re.match(regex_result_file, result_yaml.basename())
        if m is not None:
            with open(result_yaml, 'r') as f:
                data = yaml.safe_load(f)
            fname_infos = m.groupdict()
            if 'iteration' in fname_infos: fname_infos['iteration'] = int(fname_infos['iteration'])
            data = dict(**data, **fname_infos)
            table.append(data)
    table = pd.DataFrame(table)
    # print(len(table))
    # print('=' * 50)
    # print('Table iteration 100 000')
    # print(table.loc[table.iteration == 6_250])
    table = table.loc[table.groupby(gather_parameters)['avg_loss'].idxmin()]
    print('=' * 50)
    print('Table best results')
    print(table.to_string())
    print('=' * 50)
    return table


def setup_argparse():
    parser = ArgumentParser()
    parser.add_argument('-t', '--trigonometric', action='store_true', dest='trigonometric')
    parser.add_argument('-s', '--stats', action='store_true', dest='calc_stats')
    parser.add_argument('-l', '--print-latex', action='store_true', dest='print_latex')
    parser.add_argument('-f', '--float-format', dest='float_format', type=str, default='%.6f')
    return parser


def gather_results_runs_polynomials(args) -> pd.DataFrame:
    return gather_results_runs(['avg_loss', 'max_loss', 'd'], ['d'], REGEX_RESULT_FILE)


def gather_results_runs_trigonometric(args) -> pd.DataFrame:
    assert args.trigonometric
    return gather_results_runs(['avg_loss', 'max_loss', 'd', 'r'], ['d', 'r'],
                               REGEX_TRIGONOMETRIC_RESULT_FILE)


def gather_results_runs(vars, index, regex_result_file):
    tables = []
    for run_dir in RUNS_DIR.dirs():
        print('run =', str(run_dir))
        table = gather_results_run(run_dir, regex_result_file, index)
        table = table[vars]
        tables.append(table)
    tables = pd.concat(tables)
    return tables


def format_stats(stats_table, float_format='%.6f'):
    format_f = lambda x: float_format % x
    format_table = pd.DataFrame()
    format_table['avg_loss'] = (stats_table['avg_loss']['mean'].apply(format_f) + ' ± ') + stats_table[
        'avg_loss']['std'].apply(format_f).values
    format_table['max_loss'] = (stats_table['max_loss']['mean'].apply(format_f) + ' ± ') + stats_table[
        'max_loss']['std'].apply(format_f).values
    return format_table


def store_result(table, print_latex: bool, fname: str = 'result'):
    table.to_csv(f'{fname}.csv')
    table.to_markdown(f'{fname}.md')
    if print_latex:
        print(table.to_latex())


if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    if args.calc_stats:
        if args.trigonometric:
            tables = gather_results_runs_trigonometric(args)
            group_vars = ['d', 'r']
        else:
            tables = gather_results_runs_polynomials(args)
            group_vars = ['d']
        print(tables)
        stats_table = tables.groupby(group_vars).aggregate(['mean', 'std'])
        print(stats_table.to_string())
        stats_table = format_stats(stats_table, args.float_format)
        store_result(stats_table, args.print_latex)

    else:
        if args.trigonometric:
            result = gather_results_run(RUNS_DIR, REGEX_TRIGONOMETRIC_RESULT_FILE)
            store_result(result, args.print_latex, 'result_trigonometric')
        else:

            result = gather_results_run(RUNS_DIR, REGEX_RESULT_FILE)
            store_result(result, args.print_latex, 'result')
