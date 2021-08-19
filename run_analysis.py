import re
import pandas as pd
from path import Path
import numpy as np
import yaml

runs = Path(__file__).parent / 'run'

def make_table_results():
    table = []
    for run in runs.dirs():
        for result_yaml in run.files('test_result_*.yaml'):
           with open(result_yaml, 'r') as f:
                data = yaml.load(f)
           iteration = re.match(r'test_result_([0-9]+)\.yaml', result_yaml.basename()).group(1)
           iteration = int(iteration)
           data['iteration'] = iteration
           table.append(data)
    table = pd.DataFrame(table)
    print(table)
    print(table.loc[table.iteration == 10_000])
    table = table.loc[table.groupby('d')['avg_loss'].idxmin()]
    print(table.to_string())
    table.to_csv('results.csv', index=False)

if __name__ == '__main__':
    make_table_results()