import torch
import yaml
from torch.utils.data import DataLoader

from nn.model import Model
from train import run_eval
from path import Path
from nn.data import BezierRandomGenerator, TrigonometricRandomGenerator
import re
from argparse import ArgumentParser

model_regex = re.compile(r'model_([0-9]+)\.pth')
d_r_map = {2: range(2, 6), 3: range(2, 6), 4: range(3, 7), 5: range(4, 8)}
ROOT = Path(__file__).parent


def test_all_polynomials(runs_dir, device: str, test_size: int = 1_000):
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
            model.load_state_dict(torch.load(model_path, map_location=device))
            run_eval(model, device, run, dl, d, test_size, iteration, 'test_result')
            print('eval done to', str(model_path))


def test_all_trigonometric(runs_dir, device: str, test_size: int, trigonometric_start_range: float, trigonometric_end_range: float):
    d_r_map = {2: range(2, 6), 3: range(2, 6), 4: range(3, 7), 5: range(4, 8)}
    for run in runs_dir.dirs():
        with open(run / 'config.yaml') as f:
            config = yaml.safe_load(f)
        d = config['d']
        model_paths = run.files('model*.pth')
        model_paths = [(model_path, int(model_regex.match(model_path.basename()).group(1)))
                       for model_path in model_paths if model_regex.match(model_path.basename())]
        model = Model(d).to(device)
        sequence_r = d_r_map[d]
        for model_path, iteration in model_paths:
            for r in sequence_r:  # r = degree trigonometric function
                dataset = TrigonometricRandomGenerator(r, 2 * d + 1, trigonometric_start_range, trigonometric_end_range)
                dl = DataLoader(dataset, 16)
                model.load_state_dict(torch.load(model_path, map_location=device))
                try:
                    result = run_eval(model, device, run, dl, d, test_size, iteration, None)
                    result['r'] = r
                    with open(run / f'test_result_trigonometric_{iteration}_{r}.yaml', 'w') as f:
                        yaml.safe_dump(result, f)
                    print('eval done to', str(model_path))
                except RuntimeError as e:
                    print('Skipped', run, 'because RuntimeError', e.args)


def setup_argparse():
    parser = ArgumentParser()
    parser.add_argument('-t', '--trigonometric', action='store_true', dest='trigonometric')
    parser.add_argument('--trigonometric-end-range', '--trigonometric-end-value',
                        type=float, dest='trigonometric_end_range', default=1.0)
    parser.add_argument('--trigonometric-start-range', '--trigonometric-start-value',
                        type=float, dest='trigonometric_start_range', default=0.0)
    parser.add_argument('--folder', '-f', dest='folder', type=str, default='run')
    parser.add_argument('--test-size', dest='test_size', type=int, default=1_000)
    parser.add_argument('--device', '-d', dest='device', type=str, default='cpu')
    return parser


if __name__ == '__main__':
    parser = setup_argparse()
    args = parser.parse_args()
    if args.trigonometric:
        test_all_trigonometric(ROOT / args.folder, args.device, args.test_size, args.trigonometric_start_range, args.trigonometric_end_range)
    else:
        test_all_polynomials(ROOT / args.folder, args.device, args.test_size)
