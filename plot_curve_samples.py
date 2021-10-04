from argparse import ArgumentParser
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from path import Path
from torch.utils.data import DataLoader

from nn.data import BezierRandomGenerator, bezier_curve_batch, get_optimal_c, \
                    bezier_curve, TrigonometricRandomGenerator, trigonometric_fun_a_b_t
from nn.model import Model
from test import d_r_map
from train import EvalStep

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
def draw_samples(d_s: list, num_samples: int, model_paths: list, ):
    assert len(d_s) == len(model_paths), f'{len(d_s) = } - {len(model_paths) = }'
    fig, axs = plt.subplots(len(d_s), num_samples, figsize=(14, 6))
    if len(d_s) == 1: axs = np.expand_dims(axs, 0)
    for i, (d, model_path) in enumerate(zip(d_s, model_paths)):
        print(f'{d = } - {model_path = }')
        dataset = BezierRandomGenerator(d, 2 * d + 1)
        dl = DataLoader(dataset, batch_size=1)
        iter_dl = iter(dl)
        model = Model(d)
        model.eval()
        step = EvalStep(model, 'cpu')
        model.load_state_dict(torch.load(model_path, 'cpu'))
        axs[i, 0].set_ylabel(f'd = {d}')
        for j in range(num_samples):
            batch_curves = next(iter_dl)
            pred = step(batch_curves)
            p_pred = bezier_curve_batch(pred['c_hat'], torch.linspace(0, 1, 1_000).unsqueeze(0)).squeeze(0)
            fig1, ax = plt.subplots(1, 1)
            plt.scatter(batch_curves['p'][0, :, 0], batch_curves['p'][0, :, 1], c='red', label='True points')
            plt.scatter(pred['p_pred'][0, :, 0], pred['p_pred'][0, :, 1], c='blue', s=20)
            plt.plot(p_pred[:, 0], p_pred[:, 1], c='blue', label='NN curve')
            plot_closed_form_f(batch_curves, d, uniform_t, 'Uniform', '#B8BF12', axs[i, j])
            plot_closed_form_f(batch_curves, d, chordlength_t, 'Chordlength', 'orange', axs[i, j])
            plot_closed_form_f(batch_curves, d, partial(centripetal_t, a=0.5), 'Centripetal', 'green', axs[i, j])
            fname = f'{d=}_{j}.png'
            plt.legend()
            plt.savefig(PLOTS / fname)
            plt.close()
            axs[i, j].scatter(batch_curves['p'][0, :, 0], batch_curves['p'][0, :, 1], s=30.0, c='red', label='True points')
            axs[i, j].scatter(pred['p_pred'][0, :, 0], pred['p_pred'][0, :, 1], c='blue', s=20)
            axs[i, j].plot(p_pred[:, 0], p_pred[:, 1], c='blue', label='NN curve')
            axs[i, j].xaxis.set_ticks([0,1])
            axs[i, j].yaxis.set_ticks([])

    axs[0, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    fig.tight_layout()
    mosaic_fname = 'mosaic.png'
    fig.savefig(PLOTS / mosaic_fname)


@torch.no_grad()
def draw_samples_trigonometric(d_s: list, num_samples: int, model_paths: list):
    fig, axs = plt.subplots(len(d_s), num_samples * 4, figsize=(14, 7))  # 4 = num of r for each d
    if len(d_s) == 1: axs = np.expand_dims(axs, 0)
    for i, (d, model_path) in enumerate(zip(d_s, model_paths)):
        print(f'{d = } - {model_path = }')
        model = Model(d)
        model.eval()
        step = EvalStep(model, 'cpu')
        model.load_state_dict(torch.load(model_path, 'cpu'))
        axs[i, 0].set_ylabel(f'd = {d}')
        r_seq = d_r_map[d]
        for k, r in enumerate(r_seq):
            print(f'{r = }')
            dataset = TrigonometricRandomGenerator(r, 2 * d + 1)
            dl = DataLoader(dataset, batch_size=1)
            iter_dl = iter(dl)
            axs[i, k * num_samples].set_title(f'{r=}')
            for j in range(num_samples):
                jk = k * num_samples + j
                batch_curves = next(iter_dl)
                t_range = torch.linspace(0, 6.28, 10_000)
                for i in range(len(batch_curves['A'])):
                    true_curve = trigonometric_fun_a_b_t(batch_curves['A'][i], batch_curves['B'][i], batch_curves['a_0'][i], r, t_range)
                    plt.plot(true_curve['p'][:, 0], true_curve['p'][:, 1])
                    plt.title(f'{r = } in range[0, 6.28')
                    plt.show()
                pred = step(batch_curves)
                p_pred = bezier_curve_batch(pred['c_hat'], torch.linspace(0, 1, 1_000).unsqueeze(0)).squeeze(0)
                fig1, ax = plt.subplots(1, 1)
                plt.scatter(batch_curves['p'][0, :, 0], batch_curves['p'][0, :, 1], c='red', label='True points')
                plt.scatter(pred['p_pred'][0, :, 0], pred['p_pred'][0, :, 1], c='blue', s=20)
                plt.plot(p_pred[:, 0], p_pred[:, 1], c='blue', label='NN curve')
                plt.legend()
                fname = f'trigonometric_{d=}_{r=}_{j}.png'
                plt.savefig(PLOTS / fname)
                plt.close()
                axs[i, jk].scatter(batch_curves['p'][0, :, 0], batch_curves['p'][0, :, 1], s=30.0, c='red', label='True points')
                axs[i, jk].scatter(pred['p_pred'][0, :, 0], pred['p_pred'][0, :, 1], c='blue', s=20)
                axs[i, jk].plot(p_pred[:, 0], p_pred[:, 1], c='blue', label='NN curve')
                axs[i, jk].xaxis.set_ticks([])
                axs[i, jk].yaxis.set_ticks([])

    axs[0, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='x-small')
    mosaic_fname = 'trigonometric_mosaic.png'
    fig.savefig(PLOTS / mosaic_fname)


def draw_sample_trigonometric_input_only(n: int):
    for r in range(2, 8):
        print(f'{r = }')
        dataset = TrigonometricRandomGenerator(r, 10_000, -6.28, 6.28)
        iter_dataset = iter(dataset)
        for _ in range(n):
            curve = next(iter_dataset)
            p = curve['p']
            plt.plot(p[:, 0], p[:, 1])
            print(curve['t'])
            idx_t0 = curve['t'].pow(2).argmin()
            p_t0 = p[idx_t0]
            idx_t1 = (curve['t'] - 1.0).pow(2).argmin()
            p_t1 = p[idx_t1]
            print(idx_t0, idx_t1)
            plt.plot(p[idx_t0:idx_t1, 0], p[idx_t0:idx_t1,1], c='green', label='line in [0,1]')
            plt.scatter(p_t0[0], p_t0[1], c='red', label='t=0')
            plt.scatter(p_t1[0], p_t1[1], c='orange', label='t=1')
            plt.legend()
            plt.title(f'{r = } in range[-6.28, 6.28]')
            plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--polynomial-degree', type=int, dest='d', nargs='+')
    parser.add_argument('-m', '--model', type=Path, dest='model_path', nargs='+')
    parser.add_argument('-n', type=int, default=None, dest='n')
    parser.add_argument('-t', '--trigonometric', action='store_true', dest='trigonometric')
    parser.add_argument('--input-only' , action='store_true', dest='input_curve_only')

    args = parser.parse_args()
    print(f'{args = }')
    if args.input_curve_only:
        if args.trigonometric:
            draw_sample_trigonometric_input_only(args.n)
        else:
            raise NotImplementedError()
    else:
        if args.trigonometric:
            draw_samples_trigonometric(args.d, args.n, args.model_path)
        else:
            draw_samples(args.d, args.n, args.model_path)


if __name__ == '__main__':
    main()
