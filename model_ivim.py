import torch
from torch import nn

fmin = 0;   fmax = 0.2
Dmin = 0;   Dmax = 1.8e-3
Dpmin = 0;  Dpmax = 20e-3
S0min = 200;S0max = 1700

class Net(nn.Module):
    def __init__(self, num_bvalue):
        super(Net, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.dense1 = nn.Linear(num_bvalue, 20)
        self.activation1 = nn.ELU()
        self.dense2 = nn.Linear(20, 20)
        self.activation2 = nn.ELU()
        self.dense3 = nn.Linear(20, 20)
        self.activation3 = nn.ELU()
        self.dense4 = nn.Linear(20, 4)

    def forward(self, x, v, test_input, type, noise_real, noise_imag):
        if (type == 'train') or (type == 'validation'):
            f = x[:, 0].unsqueeze(1)
            D = x[:, 1].unsqueeze(1)
            Dp = x[:, 2].unsqueeze(1)
            S0 = x[:, 3].unsqueeze(1)

            v = torch.unsqueeze(v, 0)
            x = torch.cat([v] * len(x), 0)
            x = self.sigmoid(x)
            x = x * 1000
            bvalue = x

            x = S0 * (f * torch.exp(-x * Dp) + (1 - f) * torch.exp(-x * D)) + noise_real
            x = torch.sqrt(x**2+noise_imag**2)

        elif type == 'test':
            f = D = Dp = 0 #
            v = torch.unsqueeze(v, 0)
            x = test_input
            bvalue = v

        Signal = x
        x = self.activation1(self.dense1(x))
        x = self.activation2(self.dense2(x))
        x = self.activation3(self.dense3(x))

        #=====================Quantification loss========
        x = self.dense4(x)
        f_pred = x[:, 0].unsqueeze(1)*(fmax-fmin)+fmin
        D_pred = x[:, 1].unsqueeze(1)*(Dmax-Dmin)+Dmin
        Dp_pred = x[:, 2].unsqueeze(1)*(Dpmax-Dpmin)+Dpmin
        S0_pred = x[:, 3].unsqueeze(1)*(S0max-S0min)+S0min
        #================================================

        bvalue_optimized = bvalue[0, :].unsqueeze(0)
        Signal_pred = S0_pred * (f_pred * torch.exp(-bvalue * Dp_pred) + (1 - f_pred) * torch.exp(-bvalue * D_pred))


        return Signal, bvalue_optimized, x, f, D, Dp, Signal_pred
