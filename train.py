import argparse
from operator import itemgetter

import torch.nn
import yaml
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import RunningAverage, Average, VariableAccumulation
from path import Path
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import BezierRandomGenerator, solve_system_lambdas, bezier_curve_batch, get_optimal_c
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
    parser.add_argument('--num-train-curves', '--train-size', '-n', type=int, default=1_000_000, dest='train_size')
    parser.add_argument('--num-test-curves', '--test-size', '--n-eval', type=int, default=1_000, dest='test_size')
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
        self.mse = torch.nn.MSELoss(reduction='none')
        self.max_mse = torch.nn.MSELoss(reduction='none')

    def __call__(self, batch):
        for k, el in batch.items():
            if isinstance(el, torch.Tensor):
                batch[k] = el.to(self.device)
        e = batch['e']
        d1: int = batch['b'].shape[-1]  # = d + 1
        d = d1 - 1
        lambdas = self.model(e)
        t_pred = solve_system_lambdas(lambdas).squeeze(-1).to(self.device)  # [bs, n]
        c_hat = get_optimal_c(batch['p'], batch['b'], d, t_pred, e.device).to(self.device)
        p_pred = bezier_curve_batch(c_hat, t_pred)
        loss_value = self.mse(batch['p'], p_pred).sum(dim=-1).mean()
        max_loss = self.mse(batch['p'], p_pred).max()

        return dict(**batch, p_pred=p_pred, t_pred=t_pred, loss=loss_value, max_loss=max_loss, c_hat=c_hat)


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


def run_eval(model, device, run_folder, test_dl: DataLoader, d: int, test_size: int = None, iteration: int = None, fname='test_result'):
    eval_step = EvalStep(model, device)
    evaluator = Engine(lambda e, b: eval_step(b))
    Average(itemgetter('loss')).attach(evaluator, 'avg_loss')
    VariableAccumulation(max, output_transform=itemgetter('loss')).attach(evaluator, 'max_loss')
    evaluator.state.p_s = []
    evaluator.state.p_pred_s = []
    evaluator.state.losses = []

    @evaluator.on(Events.ITERATION_COMPLETED)
    def accumulate_output(engine):
        engine.state.losses.append(engine.state.output['max_loss'].item())
        engine.state.p_s.append(engine.state.output['p'])
        engine.state.p_pred_s.append(engine.state.output['p_pred'])

    model.eval()
    with torch.no_grad():
        evaluator.run(test_dl, 1, test_size)
    avg_loss = evaluator.state.metrics['avg_loss']
    evaluator.state.p_s = torch.cat(evaluator.state.p_s, dim=0)
    evaluator.state.p_pred_s = torch.cat(evaluator.state.p_pred_s, dim=0)
    hausdorff_error = max(torch.min((x - evaluator.state.p_s).pow(2)).item() for x in evaluator.state.p_pred_s)
    evaluator.state.metrics['hausdorff_loss'] = hausdorff_error
    model.train()
    print(f'avg loss for d = {d} is {avg_loss}')
    result = dict(d=d, avg_loss=avg_loss,
                  max_loss=max(evaluator.state.losses),
                  hausdorff_loss=hausdorff_error)
    if fname:
        dst_path = run_folder / f'{fname}_{iteration}.yaml' if iteration else run_folder / f'{fname}.yaml'
        with open(dst_path, 'w') as f:
            yaml.dump(result, f)
    return result


def setup_logger(engine, model, opt, run_folder):
    logger = SummaryWriter(run_folder)

    @engine.on(Events.ITERATION_COMPLETED(every=1_000))
    def log_loss(engine: Engine):
        logger.add_scalar('train/loss', engine.state.output['loss'].item(), engine.state.iteration)
        logger.add_scalar('train/running_avg_loss', engine.state.metrics['running_avg_loss'], engine.state.iteration)

    @engine.on(Events.ITERATION_COMPLETED(every=1_000))
    def log_model_params(engine: Engine):
        norm_params = sum(p.norm(1).item() for p in model.parameters())
        norm_grad = sum(p.grad.norm(1).item() for p in model.parameters() if p.grad is not None)
        logger.add_scalar('model/norm_params', norm_params, engine.state.iteration)
        logger.add_scalar('model/norm_params_grad', norm_grad, engine.state.iteration)

    @engine.on(Events.ITERATION_COMPLETED(every=1_000))
    def log_lr(engine: Engine):
        lrs = [param_group['lr'] for param_group in opt.param_groups]
        avg_lr = sum(lrs) / len(lrs)
        logger.add_scalar('opt/avg_lr', avg_lr, engine.state.iteration)


def train(args, run_folder):
    model = Model(args.d).to(args.device)
    opt = Adam(model.parameters(), args.lr)
    n = 2 * args.d + 1
    dataset = BezierRandomGenerator(args.d, n)
    eval_dataset = BezierRandomGenerator(args.d, n)
    dl = DataLoader(dataset, batch_size=args.batch_size)
    eval_dl = DataLoader(eval_dataset, batch_size=args.batch_size)
    step = TrainStep(model, args.device, opt)
    trainer = Engine(lambda e, b: step(b))

    RunningAverage(output_transform=itemgetter('loss')).attach(trainer, 'running_avg_loss')
    ProgressBar().attach(trainer, ['running_avg_loss'])
    es = EarlyStopping(5, lambda e: -e.state.metrics['running_avg_loss'], trainer)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=5_000), es)

    @trainer.on(Events.ITERATION_COMPLETED(once=6_250) | Events.ITERATION_COMPLETED(every=5_000))
    def training_eval(engine):
        return run_eval(model, args.device, run_folder, eval_dl, args.d, args.test_size, engine.state.iteration)

    setup_logger(trainer, model, opt, run_folder)

    @trainer.on(Events.ITERATION_COMPLETED(every=5_000))
    def save_model(engine):
        torch.save(model.state_dict(), run_folder / f'model_{engine.state.iteration}.pth')

    trainer.run(dl, max_epochs=1, epoch_length=args.train_size)

    torch.save(model.state_dict(), run_folder / 'model.pth')


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
