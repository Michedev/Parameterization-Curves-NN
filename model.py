from torch import nn
from torch.nn.functional import linear, batch_norm
class ResBlock(nn.Module):

    def __init__(self, n_in, n_hid, n_out):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out
        self.main_road = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.BatchNorm1d(n_hid),  # warning: dimension problem because prev layer output is [n, 2, n_hid] but batch norm needs n_hid in second dimension
            nn.ReLU(),
            nn.Linear(n_hid, n_hid),
            nn.BatchNorm1d(n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, n_out),
            nn.BatchNorm1d(n_out)
        )
        self.skip_road = None
        if n_in != n_out:
            self.skip_road = nn.Sequential(
                nn.Linear(n_in, n_out),
                nn.BatchNorm1d(n_out)
            )

    def forward(self, x):
        out_main = self.main_road(x)
        if self.skip_road:
            out_main = out_main + self.skip_road(x)
        return out_main.relu()


class Model(nn.Module):

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.block_1 = nn.Sequential(
            nn.Linear(4 * d, 30),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            ResBlock(30, 50, 70),
            ResBlock(70, 70, 70),
            ResBlock(70, 70, 70),
            ResBlock(70, 50, 30),
            ResBlock(30, 30, 30)
        )

        self.block_end = nn.Sequential(
            nn.Linear(30, 4 * d - 2),
        )


    def forward(self, x):
        bs = len(x)
        x_out = self.block_1(x.flatten(1))
        x_out = self.res_blocks(x_out)
        x_out = self.block_end(x_out).view(bs, 2 * self.d -1, 2)
        x_out = x_out.softmax(dim=-1)
        return x_out