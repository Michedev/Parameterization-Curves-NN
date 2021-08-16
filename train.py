import argparse
from operator import itemgetter

import torch.nn
import yaml
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage, Average, MetricsLambda
from path import Path
from ignite.engine import Engine, Events
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import BezierRandomGenerator, solve_system_lambdas, bezier_curve_batch
from model import Model

ROOT = Path(__file__).parent
RUN = ROOT / 'run'
if not RUN.exists():
    RUN.mkdir()
MODEL_PATH: Path = ROOT / 'model.py'


def get_new_experiment_folder():
    i = 0
    while (RUN / str(i)).exists():
        i += 1
    run_folder = RUN / str(i)
    run_folder.mkdir()
    return run_folder


def setup_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=int, required=True, dest='d')
    parser.add_argument('--num-curves', '-n', type=int, default=1_000_000, dest='n')
    parser.add_argument('--num-curves-eval', '--n-eval', type=int, default=1_000, dest='n_eval')
    parser.add_argument('--batch-size', '-b', type=int, dest='batch_size', default=16)
    parser.add_argument('--lr', '-l', type=float, dest='lr', default=1e-3)
    parser.add_argument('--seed', '-s', type=int, dest='seed', default=13)
    parser.add_argument('--device', type=str, dest='device', default='cpu')

    return parser


class EvalStep:

    def __init__(self, model, device, opt=None):
        self.model = model
        self.opt = opt
        self.device = device
        self.mse = torch.nn.MSELoss()

    def __call__(self, batch):
        e = batch['e']
        d1: int = batch['b'].shape[-1]
        d = d1 - 1
        lambdas = self.model(e)
        t_pred = solve_system_lambdas(lambdas).squeeze(-1)  # [bs, n]
        T_pred = torch.zeros(*t_pred.shape, d1, device=e.device)
        T1_pred = torch.zeros(*t_pred.shape, d1, device=e.device)
        for j in range(d1):
            T_pred[:, :, j] = t_pred.pow(j)
            T1_pred[:, :, j] = (1 - t_pred).pow(d - j)
        A: torch.FloatTensor = T_pred * T1_pred * batch['b']
        A_T = A.transpose(1, 2)
        c_hat = torch.inverse(A_T @ A) @ A_T @ batch['p']
        p_pred = bezier_curve_batch(c_hat, t_pred)
        loss_value = self.mse(batch['p'], p_pred)
        return dict(**batch, p_pred=p_pred, t_pred=t_pred, loss=loss_value)


class TrainStep:

    def __init__(self, model, device, opt):
        self.opt = opt
        self.eval_step = EvalStep(model, device)

    def __call__(self, batch):
        self.opt.zero_grad()
        eval_output = self.eval_step(batch)
        eval_output['loss'].backward()
        self.opt.step()
        return eval_output


def test(model, device, run_folder, test_dl: DataLoader, d: int):
    eval_step = EvalStep(model, device)
    evaluator = Engine(lambda e, b: eval_step(b))
    Average(itemgetter('loss')).attach(evaluator, 'avg_loss')
    model.eval()
    with torch.no_grad():
        evaluator.run(test_dl, 1)
    avg_loss = evaluator.state.metrics['avg_loss']
    model.train()
    print(f'avg loss for d = {d} is {avg_loss}')
    result = dict(d=d, avg_loss=avg_loss)
    with open(run_folder / 'test_result.yaml', 'w') as f:
        yaml.dump(result, f)
    return result


def train(args, run_folder):
    model = Model(args.d).to(args.device)
    opt = Adam(model.parameters(), args.lr)
    dataset = BezierRandomGenerator(args.d, args.n)
    eval_dataset = BezierRandomGenerator(args.d, args.n_eval)
    dl = DataLoader(dataset, batch_size=args.batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=args.batch_size)
    step = TrainStep(model, args.device, opt)
    trainer = Engine(lambda e, b: step(b))

    RunningAverage(output_transform=itemgetter('loss')).attach(trainer, 'running_avg_loss')
    ProgressBar().attach(trainer, ['running_avg_loss'])

    trainer.add_event_handler(Events.EPOCH_COMPLETED, test, model, args.device, run_folder, eval_dl, args.d)

    trainer.run(dl, max_epochs=1)

    torch.save(model.parameters(), run_folder / 'model.pth')


def main():
    run_folder: Path = get_new_experiment_folder()
    parser = setup_argparser()
    args = parser.parse_args()
    with open(run_folder / 'config.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)
    MODEL_PATH.copy(run_folder / 'model.py')
    train(args, run_folder)


if __name__ == '__main__':
    main()
