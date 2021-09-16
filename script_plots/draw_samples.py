from argparse import ArgumentParser
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from path import Path
from torch.utils.data import DataLoader

from data import BezierRandomGenerator, bezier_curve_batch, get_optimal_c, bezier_curve, scale_points, \
    TrigonometricRandomGenerator, trigonometric_fun, trigonometric_fun_t
from model import Model
from train import TrainStep, EvalStep

PLOTS = Path(__file__).parent.parent / 'plots'
if not PLOTS.exists():
    PLOTS.mkdir()


def uniform_t(p):
    n = p.shape[0]
    return torch.arange(n) / (n - 1)


def centripetal_t(p, a: float):
    result = torch.zeros(len(p))
    den = 0.0
    num = 0.0
    for i in range(1, len(p)): den += (p[i] - p[i - 1]).norm(2) ** a
    for i in range(1, len(p) - 1):
        num += (p[i] - p[i - 1]).norm(2) ** a
        result[i] = num / den
    result[-1] = 1.0
    result[0] = 0.0
    return result


def chordlength_t(p):
    return centripetal_t(p, a=1)


def plot_closed_form_f(curve: dict, d: int, t_func, t_func_name: str, color='yellow', ax=None):
    p = curve['p']
    t = t_func(p.squeeze(0)).unsqueeze(0)
    c = get_optimal_c(p, curve['b'], d, t, 'cpu').squeeze(0)
    t = t.squeeze(0)
    p_pred = bezier_curve(c, t).squeeze(0)
    p_pred_1000 = bezier_curve(c, torch.linspace(0, 1, 1_000))
    plt.scatter(p_pred[:, 0], p_pred[:, 1], c=color, s=5)
    plt.plot(p_pred_1000[:, 0], p_pred_1000[:, 1], '--', c=color, label=f'{t_func_name}')
    if ax:
        ax.scatter(p_pred[:, 0], p_pred[:, 1], c=color)
        ax.plot(p_pred_1000[:, 0], p_pred_1000[:, 1], '--', c=color, label=f'{t_func_name}')
    # plt.scatter(c[:, 0], c[:, 1], marker='D', c=color, label=f'{t_func_name} control points')
    # plt.plot(c[:, 0], c[:, 1], '--', c=color, label=f'{t_func_name} control polygon')

def plot_closed_form_f_trigonoetric(curve: dict, d: int, t_func, t_func_name: str, color='yellow', ax=None):
    p = curve['p']
    t = t_func(p.squeeze(0)).unsqueeze(0)
    c = get_optimal_c(p, curve['b'], d, t, 'cpu').squeeze(0)
    t = t.squeeze(0)
    p_pred = bezier_curve(c, t).squeeze(0)
    p_pred_1000 = bezier_curve(c, torch.linspace(0, 1, 1_000))
    plt.scatter(p_pred[:, 0], p_pred[:, 1], c=color, s=5)
    plt.plot(p_pred_1000[:, 0], p_pred_1000[:, 1], '--', c=color, label=f'{t_func_name}')
    if ax:
        ax.scatter(p_pred[:, 0], p_pred[:, 1], c=color)
        ax.plot(p_pred_1000[:, 0], p_pred_1000[:, 1], '--', c=color, label=f'{t_func_name}')
    # plt.scatter(c[:, 0], c[:, 1], marker='D', c=color, label=f'{t_func_name} control points')
    # plt.plot(c[:, 0], c[:, 1], '--', c=color, label=f'{t_func_name} control polygon')



@torch.no_grad()
def draw_samples(d_s: list, num_samples: int, model_paths: list, trigonometric: bool = False, trigonometric_degrees: list = None):
    if trigonometric_degrees is None:
        trigonometric_degrees = d_s[:]
    assert len(d_s) == len(model_paths), f'{len(d_s) = } - {len(model_paths) = }'
    fig, axs = plt.subplots(len(d_s), num_samples, figsize=(14, 6))
    if len(d_s) == 1: axs = np.expand_dims(axs, 0)
    for i, (d, model_path) in enumerate(zip(d_s, model_paths)):
        print(f'{d = } - {model_path = }')
        if trigonometric:
            dataset = TrigonometricRandomGenerator(trigonometric_degrees[i], 2 * d + 1)
        else:
            dataset = BezierRandomGenerator(d, 2 * d + 1)
        dl = DataLoader(dataset, batch_size=1)
        iter_dl = iter(dl)
        model = Model(d)
        model.eval()
        step = EvalStep(model, 'cpu')
        model.load_state_dict(torch.load(model_path))
        axs[i, 0].set_ylabel(f'd = {d}')
        for j in range(num_samples):
            batch_curves = next(iter_dl)
            pred = step(batch_curves)
            p_pred = bezier_curve_batch(pred['c_hat'], torch.linspace(0, 1, 1_000).unsqueeze(0)).squeeze(0)
            fig1, ax = plt.subplots(1, 1)
            plt.scatter(batch_curves['p'][0, :, 0], batch_curves['p'][0, :, 1], c='red', label='True points')
            plt.scatter(pred['p_pred'][0, :, 0], pred['p_pred'][0, :, 1], c='blue', s=5)
            plt.plot(p_pred[:, 0], p_pred[:, 1], c='blue', label='NN curve')
            plot_closed_form_f(batch_curves, d, uniform_t, 'Uniform', '#B8BF12', axs[i, j])
            plot_closed_form_f(batch_curves, d, chordlength_t, 'Chordlength', 'orange', axs[i, j])
            plot_closed_form_f(batch_curves, d, partial(centripetal_t, a=0.5), 'Centripetal', 'green', axs[i, j])
            fname = f'{d=}_{j}.png'
            if trigonometric:
                fname = 'trigonometric_' + fname
            plt.savefig(PLOTS / fname)
            plt.close()
            axs[i, j].scatter(batch_curves['p'][0, :, 0], batch_curves['p'][0, :, 1], s=30.0, c='red', label='True points')
            axs[i, j].scatter(pred['p_pred'][0, :, 0], pred['p_pred'][0, :, 1], c='blue', s=5)
            axs[i, j].plot(p_pred[:, 0], p_pred[:, 1], c='blue', label='NN curve')
    axs[0, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    fig.tight_layout()
    mosaic_fname = 'mosaic.png'
    if trigonometric:
        mosaic_fname = 'trigonometric_' + mosaic_fname
    fig.savefig(PLOTS / mosaic_fname)


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--polynomial-degree', type=int, dest='d', nargs='+')
    parser.add_argument('-m', '--model', type=Path, dest='model_path', nargs='+')
    parser.add_argument('-n', type=int, default=None, dest='n')
    parser.add_argument('-t', '--trigonometric', action='store_true', dest='trigonometric')
    # parser.add_argument('-r', '--trigonometric-degree', type=int, nargs='+', dest='r')

    args = parser.parse_args()
    print(f'{args = }')
    draw_samples(args.d, args.n, args.model_path, args.trigonometric)


if __name__ == '__main__':
    main()
