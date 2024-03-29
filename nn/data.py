from typing import Iterator

import torch
from scipy.special import binom
from torch.utils.data import IterableDataset


def bezier_curve(c, t):
    """
    :param c: vector of control points with shape [d+1, 2]
    :param t: vector of input parameters with shape [n]
    :return: the value of the curve at the parameters t
    """
    assert c.device == t.device
    n = t.shape[0]
    d1 = c.shape[0]  # = d + 1
    d = d1 - 1
    one_min_t = 1 - t
    T = torch.zeros(n, d1, device=c.device)
    T1 = torch.zeros(n, d1, device=c.device)
    B = torch.zeros(d1, 1, device=c.device)
    for j in range(d1):
        T[:, j] = t ** j
        T1[:, j] = one_min_t ** (d - j)
        B[j] = binom(d, j)
    P = torch.matmul(T * T1, B * c)
    return P


def bezier_curve_batch(c, t):
    """
    Batched version of the previous function.
    :param c: vector of control points with shape [bs, d, 2]
    :param t: vector of input parameters with shape [bs, n]
    :return: the value of the curve at the parameters t
    """
    assert c.device == t.device
    t = t.squeeze(-1)
    bs = t.shape[0]
    n = t.shape[1]
    d1 = c.shape[1]
    d = d1 - 1
    one_min_t = 1 - t
    T = torch.zeros(bs, n, d1, device=c.device)
    T1 = torch.zeros(bs, n, d1, device=c.device)
    B = torch.zeros(1, d1, 1, device=c.device)
    for j in range(d1):
        T[:, :, j] = t ** j
        T1[:, :, j] = one_min_t ** (d - j)
        B[0, j] = binom(d, j)
    P = torch.matmul(T * T1, B * c)
    return P


def generate_curve(d: int, n : int = None):
    if n is None:
        n = 2 * d + 1
    c = torch.randn(d + 1, 2)
    t = torch.rand(n).sort()[0]
    t[0] = 0
    t[-1] = 1
    return c, t


def scale_points(p):
    min_p = [torch.min(p[:, 0]), torch.min(p[:, 1])]
    max_p = (p[:, 0].max() - min_p[0], p[:, 1].max() - min_p[1])
    max_p = max(max_p)
    min_p = torch.FloatTensor(min_p).to(p.device).unsqueeze(0)
    p1 = (p - min_p) / max_p
    return p1


def trigonometric_fun(d: int, n: int, device = 'cpu'):
    """
    Compute f(t_i) = a_0 + \sum_{j=1}^{d}  a_j cos(jt) + i \sum_{j=1}^{d} b_j sin(jt)  i=1...n
    a_0, a_j, b_j are sampled from N(0,1)
    t_i are sampled from U(0,1) with t_0 = 0 and t_n = 1
    :param d: degree of trigonometric function
    :param n: number of parameters
    :return: data points evaluated at t_i with shape [n, 2]
    """
    t = torch.rand(n, device=device)
    t[0] = 0.0
    t[-1] = 1.0
    return trigonometric_fun_t(d, t)

def trigonometric_fun_t(d: int, t: torch.Tensor):
    """
    Compute f(t_i) = a_0 + \sum_{j=1}^{d}  a_j cos(jt) + i \sum_{j=1}^{d} b_j sin(jt)  i=1...n
    a_0, a_j, b_j are sampled from N(0,1)
    t_i must be in range [0,1]
    :param d: degree of trigonometric function
    :param t: array of parameters with shape [n]
    :return: data points evaluated at t_i with shape [n, 2]
    """

    assert len(t.shape) == 1
    a_0 = torch.randn(1, 2, device=t.device)
    A = torch.randn(d, 2, device=t.device)
    B = torch.randn(d, 2, device=t.device)
    return trigonometric_fun_a_b_t(A, B, a_0, d, t)


def trigonometric_fun_a_b_t(A, B, a_0, d, t):
    n = len(t)
    A = A.unsqueeze(-1)
    B = B.unsqueeze(-1)
    assert len(a_0.shape) == 2, a_0.shape
    assert len(A.shape) == 3 and len(B.shape) == 3, (A.shape, B.shape)
    j = torch.arange(d, device=t.device) + 1
    j = j.view(d, 1, 1)
    t = t.view(1, 1, -1)
    t = torch.sort(t).values
    cos_jt = torch.cos(j * t)
    sin_jt = torch.sin(j * t)
    i = torch.arange(n, device=t.device) + 1
    i = i.unsqueeze(-1)   # [n, 1]
    A_cos_jt = A * cos_jt  # [d, 2, n]
    B_sin_jt = B * sin_jt  # [d, 2, n]
    A_cos_jt = A_cos_jt.sum(dim=0).permute(1, 0)  # [n, 2]
    B_sin_jt = B_sin_jt.sum(dim=0).permute(1, 0)  # [n, 2]
    p = a_0 + A_cos_jt + i * B_sin_jt
    return dict(p=p, t=t, A=A, B=B, a_0=a_0)


class BezierRandomGenerator(IterableDataset):

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.sample_points_bezier()

    def sample_points_bezier(self):
        c, t = generate_curve(self.d, self.n)
        p: torch.Tensor = bezier_curve(c, t)
        p1 = scale_points(p)
        e = torch.zeros(p1.shape[0] - 1, 2)
        for i in range(p1.shape[0] - 1):
            e[i] = p1[i + 1] - p1[i]
        return dict(p=p1, e=e, c=c, b=self.coef_bins)

    def __init__(self, d: int, n: int):
        self.d = d
        self.n = n
        self.coef_bins = torch.FloatTensor([binom(d, i) for i in range(d + 1)]).view(1, d + 1)

    def __len__(self):
        return self.n



class TrigonometricRandomGenerator(IterableDataset):


    def __init__(self, d: int, n: int, min_t_value: float = 0.0, max_t_value: float = 1.0):
        self.d = d
        self.n = n
        self.min_t_value = min_t_value
        self.max_t_value = max_t_value
        self.coef_bins = torch.FloatTensor([binom(d, i) for i in range(d + 1)]).view(1, d + 1)

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield self.sample_trigonometric_points()

    def sample_trigonometric_points(self) -> dict:
        t = torch.rand(self.n) *(self.max_t_value - self.min_t_value) + self.min_t_value
        t[0] = self.min_t_value; t[-1] = self.max_t_value
        t = torch.sort(t).values
        data = trigonometric_fun_t(self.d, t)
        p: torch.Tensor = data['p']
        p1 = scale_points(p)
        e = torch.zeros(p1.shape[0] - 1, 2)
        for i in range(p1.shape[0] - 1):
            e[i] = p1[i + 1] - p1[i]
        return dict(p=p1, e=e, b=self.coef_bins, t=t)


def solve_system_lambdas(lambdas):
    """

    :param lambdas: output of nn lambda 1 and lambda 2 with shape [bs, 2d-1]
    :return: the parameters t with shape [bs, 2d+1]
    """
    bs = lambdas.shape[0]
    n = lambdas.shape[1] + 2
    b = torch.zeros(bs, n, 1)
    A = torch.zeros(bs, n, n)
    A[:, 0, 0] = 1
    A[:, -1, -1] = 1
    b[:, -1] = 1
    for i in range(1, n - 1):
        A[:, i, i - 1] = lambdas[:, i - 1, 0]
        A[:, i, i] = -1
        A[:, i, i + 1] = lambdas[:, i - 1, 1]
    t = torch.inverse(A) @ b
    return t


def get_optimal_c(p: torch.Tensor, coef_bins: torch.Tensor, d: int, t_pred: torch.Tensor, device: str):
    d1 = d + 1
    T_pred = torch.zeros(*t_pred.shape, d1,
                         device=device)  # [bs, n, d+1], T[_, i, j] = t_i ** j (first dim is ignored because is batch dimension)
    T1_pred = torch.zeros(*t_pred.shape, d1,
                          device=device)  # [bs, n, d+1], T[_, i, j] = (1 -t_i) ** (d-j) (first dim is ignored because is batch dimension)
    for j in range(d1):
        T_pred[:, :, j] = t_pred.pow(j)
        T1_pred[:, :, j] = (1 - t_pred).pow(d - j)
    A: torch.FloatTensor = T_pred * T1_pred * coef_bins
    A_T = A.transpose(1, 2)
    c_hat = torch.inverse(A_T @ A) @ A_T @ p
    return c_hat
