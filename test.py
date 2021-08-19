import torch
import yaml
from torch.utils.data import DataLoader

from model import Model
from train import eval, RUN
from path import Path
from data import BezierRandomGenerator
import re

model_regex = re.compile(r'model_([0-9]+)\.pth')


def test_all():
    for run in RUN.dirs():
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
            eval(model, 'cpu', run, dl, d, iteration)
            print('eval done to', str(model_path))


if __name__ == '__main__':
    test_all()