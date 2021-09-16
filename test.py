import torch
import yaml
from torch.utils.data import DataLoader

from model import Model
from train import run_eval
from path import Path
from data import BezierRandomGenerator, TrigonometricRandomGenerator
import re
from argparse import ArgumentParser

model_regex = re.compile(r'model_([0-9]+)\.pth')
ROOT = Path(__file__).parent


def test_all_polynomials(runs_dir, test_size: int = 1_000):
    for run in runs_dir.dirs():
        with open(run / 'config.yaml') as f:
            config = yaml.load(f)
        d = config['d']
        dataset = BezierRandomGenerator(d, 2 * d + 1)
        dl = DataLoader(dataset, 16)
        model_paths = run.files('model*.pth')
        model_paths = [(model_path, int(model_regex.match(model_path.basename()).group(1)))
                       for model_path in model_paths if model_regex.match(model_path.basename())]
        model = Model(d)
        for model_path, iteration in model_paths:
            model.load_state_dict(torch.load(model_path))
            run_eval(model, 'cpu', run, dl, d, test_size, iteration, 'test_result')
            print('eval done to', str(model_path))

def test_all_trigonometric(runs_dir, test_size: int):
    d_r_map = {2: range(2,6), 3: range(2,6), 4: range(3,7), 5: range(4,8)}
    for run in runs_dir.dirs():
        with open(run / 'config.yaml') as f:
            config = yaml.load(f)
        d = config['d']
        model_paths = run.files('model*.pth')
        model_paths = [(model_path, int(model_regex.match(model_path.basename()).group(1)))
                       for model_path in model_paths if model_regex.match(model_path.basename())]
        model = Model(d)
        sequence_r = d_r_map[d]
        for model_path, iteration in model_paths:
            for r in sequence_r:  # r = degree trigonometric function
                dataset = TrigonometricRandomGenerator(d, 2 * d + 1)
                dl = DataLoader(dataset, 16)
                model.load_state_dict(torch.load(model_path))
                result = run_eval(model, 'cpu', run, dl, d, test_size, iteration, None)
                result['r'] = r
                with open(run / f'test_result_trigonometric_{iteration}_{r}.yaml', 'w') as f:
                    yaml.safe_dump(result, f)
                print('eval done to', str(model_path))


def setup_argparse():
    parser = ArgumentParser()
    parser.add_argument('-t', '--trigonometric', action='store_true', dest='trigonometric')
    parser.add_argument('--folder', '-f', dest='folder', type=str, default='run')
    parser.add_argument('--test-size', dest='test_size', type=int, default=1_000)
    return parser

if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    if args.trigonometric:
        test_all_trigonometric(ROOT / args.folder, args.test_size)
    else:
        test_all_polynomials(ROOT / args.folder, args.test_size)