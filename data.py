import math

import torch
from scipy.special import binom
from torch.utils.data import Dataset


def bezier_curve(c, t):
    """
    :param c: vector of control points with shape [d+1, 2]
    :param t: vector of input parameters with shape [n]
    :return: the value of the curve at the parameters t
    """
    n = t.shape[0]
    d1 = c.shape[0]  # = d + 1
    d = d1 - 1
    one_min_t = 1 - t
    T = torch.zeros(n, d1)
    T1 = torch.zeros(n, d1)
    B = torch.zeros(d1, 1)
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
    t = t.squeeze(-1)
    bs = t.shape[0]
    n = t.shape[1]
    d1 = c.shape[1]
    d = d1 - 1
    one_min_t = 1 - t
    T = torch.zeros(bs, n, d1)
    T1 = torch.zeros(bs, n, d1)
    B = torch.zeros(1, d1, 1)
    for j in range(d1):
        T[:, :, j] = t ** j
        T1[:, :, j] = one_min_t ** (d - j)
        B[0, j] = binom(d, j)
    P = torch.matmul(T * T1, B * c)
    return P


def generate_curve(d: int):
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
    min_p = torch.FloatTensor(min_p).unsqueeze(0)
    p1 = (p - min_p) / max_p
    return p1


def scholz_fun(d: int, n: int):
    a_0 = torch.randn(1, 2)
    A = torch.randn(d, 2, 1)
    B = torch.randn(d, 2, 1)
    j = torch.arange(d) + 1
    j = j.view(d, 1, 1)
    t = torch.rand(1, 1, n)
    t[0] = 0
    t[-1] = 1.0
    t = torch.sort(t).values
    cos_jt = torch.cos(j * t)
    sin_jt = torch.sin(j * t)
    i = torch.arange(n) + 1
    i = i.unsqueeze(-1)   # [n, 1]
    A_cos_jt = A * cos_jt  #[d, 2, n]
    B_cos_jt = B * cos_jt  #[d, 2, n]
    A_cos_jt = A_cos_jt.sum(dim=0).permute(1, 0)  # [n, 2]
    B_cos_jt = B_cos_jt.sum(dim=0).permute(1, 0)  # [n, 2]
    p = a_0 + A_cos_jt + i * B_cos_jt
    return dict(p=p, t=t, A=A, B=B)


class BezierRandomGenerator(Dataset):

    def __init__(self, d: int, n: int):
        self.d = d
        self.n = n
        self.coef_bins = torch.FloatTensor([binom(d, i) for i in range(d + 1)]).view(1, d + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        c, t = generate_curve(self.d)
        p: torch.Tensor = bezier_curve(c, t)
        p1 = scale_points(p)
        e = torch.zeros(p1.shape[0] - 1, 2)
        for i in range(p1.shape[0] - 1):
            e[i] = p1[i + 1] - p1[i]
        return dict(p=p1, e=e, c=c, b=self.coef_bins)


class TrigonometricRandomGenerator(Dataset):

    def __init__(self, d: int, n: int):
        self.d = d
        self.n = n
        self.coef_bins = torch.FloatTensor([binom(d, i) for i in range(d + 1)]).view(1, d + 1)


    def __len__(self):
        return self.n

    def __getitem__(self, item):
        data = scholz_fun(self.d, self.n)
        p: torch.Tensor = data['p']
        p1 = scale_points(p)
        e = torch.zeros(p1.shape[0] - 1, 2)
        for i in range(p1.shape[0] - 1):
            e[i] = p1[i + 1] - p1[i]
        return dict(p=p1, e=e, b=self.coef_bins)


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