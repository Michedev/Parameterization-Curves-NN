import torch
import yaml
from torch.utils.data import DataLoader

from model import Model
from train import run_eval, RUN
from path import Path
from data import BezierRandomGenerator, TrigonometricRandomGenerator
import re
from argparse import ArgumentParser

model_regex = re.compile(r'model_([0-9]+)\.pth')


def test_all(args):
    for run in RUN.dirs():
        with open(run / 'config.yaml') as f:
            config = yaml.load(f)
        d = config['d']
        if args.trigonometric:
            dataset = TrigonometricRandomGenerator(d, 2 * d + 1)
        else:
            dataset = BezierRandomGenerator(d, 2 * d + 1)
        dl = DataLoader(dataset, 16)
        model_paths = run.files('model*.pth')
        model_paths = [(model_path, int(model_regex.match(model_path.basename()).group(1)))
                       for model_path in model_paths if model_regex.match(model_path.basename())]
        model = Model(d)
        for model_path, iteration in model_paths:
            model.load_state_dict(torch.load(model_path))
            run_eval(model, 'cpu', run, dl, d, iteration, 'test_result_trigonometric')
            print('eval done to', str(model_path))


def setup_argparse():
    parser = ArgumentParser()
    parser.add_argument('-t', '--trigonometric', action='store_true', dest='trigonometric')
    return parser

if __name__ == '__main__':
    parser = setup_argparse()
    test_all(parser.parse_args())