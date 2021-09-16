import re
from argparse import ArgumentParser
from typing import Optional, List

import pandas as pd
from path import Path
import numpy as np
import yaml


ROOT: Path = Path(__file__).parent.abspath()
RUNS_DIR: Path = ROOT / 'run'

regex_trigonometric_result_file = r'test_result_trigonometric_([0-9]+)_([0-9]+)\.yaml'


def avg_tables(tables):
    result = tables[0].copy(deep=True)
    for i in range(1, len(tables)):
        result = result + tables[i]
    result = result / len(tables)
    return result


def gather_results_run(runs_path, regex_result_file: str = r'test_result_([0-9]+)\.yaml', gather_parameters=None) -> pd.DataFrame:
    if gather_parameters is None:
        gather_parameters = ['d']
    table = []
    for run in runs_path.dirs():
        for result_yaml in run.files('test_result_*.yaml'):
            m = re.match(regex_result_file, result_yaml.basename())
            if m:
                with open(result_yaml, 'r') as f:
                    data = yaml.safe_load(f)
                iteration = m.group(1)
                iteration = int(iteration)
                data['iteration'] = iteration
                table.append(data)
    table = pd.DataFrame(table)
    print(len(table))
    print('=' * 50)
    print('Table iteration 100 000')
    print(table.loc[table.iteration == 6_250])
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


def gather_results_runs_polynomials(args) -> List[pd.DataFrame]:
    tables = []
    for runs_dir in ROOT.dirs('run*'):
        print('run =', str(runs_dir))
        table = gather_results_run(runs_dir)
        table = table[['avg_loss', 'max_loss', 'd']]
        table = table.set_index('d')
        tables.append(table)
    return tables

def gather_results_runs_trigonometric(args) -> List[pd.DataFrame]:
    assert args.trigonometric
    tables = []
    for runs_dir in ROOT.dirs('run*'):
        print('run =', str(runs_dir))
        table = gather_results_run(runs_dir, regex_trigonometric_result_file, ['d', 'r'])
        table = table[['avg_loss', 'max_loss', 'd', 'r']]
        table = table.set_index(['d', 'r'])
        tables.append(table)
    return tables



def calc_mean_std(tables):
    avg_table = avg_tables(tables)
    std_tables = [(t - avg_table).pow(2) for t in tables]
    std_table = np.sqrt(avg_tables(std_tables))
    stats_table = pd.merge(avg_table, std_table, suffixes=('_mean', '_std'), left_index=True, right_index=True)
    return stats_table


def format_stats(stats_table, float_format = '%.6f'):
    format_f = lambda x: float_format % x
    stats_table['avg_loss'] = (stats_table['avg_loss_mean'].apply(format_f) + ' ± ') + stats_table[
        'avg_loss_std'].apply(format_f)
    stats_table['max_loss'] = (stats_table['max_loss_mean'].apply(format_f) + ' ± ') + stats_table[
        'max_loss_std'].apply(format_f)
    stats_table = stats_table[['avg_loss', 'max_loss']]
    return stats_table


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
        else:
            tables = gather_results_runs_polynomials(args)
        stats_table = calc_mean_std(tables)
        print(stats_table.to_string())
        stats_table = format_stats(stats_table, args.float_format)
        print(stats_table.to_string())
        store_result(stats_table, args.print_latex)

    else:
        if args.trigonometric:
            result = gather_results_run(RUNS_DIR, regex_trigonometric_result_file)
            store_result(result, args.print_latex, 'result_trigonometric')
        else:
            result = gather_results_run(RUNS_DIR)
            store_result(result, args.print_latex, 'result')
