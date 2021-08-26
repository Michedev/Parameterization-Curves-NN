# Implementation of "Parameterization for polynomial curve approximation via residual deep neural networks"

- Implemented using Pytorch
- To have the same environment with anaconda prompt the command `conda env create -f environment.yaml`
- When start training create a folder run/0 where there are tensorboard logs, model parameters and model code

## Table of Results

### Table 1 curve approx error with grade d and 2d + 1 points
|    avg_loss |   d |    max_loss |   iteration |
|------------:|----:|------------:|------------:|
| 0.000120546 |   2 | 0.00069892  |       40000 |
| 5.63576e-05 |   3 | 0.000901097 |       35000 |
| 6.04359e-05 |   4 | 0.000982627 |      100000 |
| 4.79376e-05 |   5 | 0.000718459 |       95000 |
### Table 3 trigonometric approx error on curves with grade d and 2d + 1 points 
|    avg_loss |   d |    max_loss |   iteration |
|------------:|----:|------------:|------------:|
| 2.17922e-05 |   2 | 0.000110496 |        5000 |
| 7.83529e-06 |   3 | 0.000100452 |       55000 |
| 8.54026e-06 |   4 | 5.07948e-05 |      100000 |
| 2.30224e-05 |   5 | 0.000197198 |       25000 |