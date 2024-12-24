import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_feats, n_hiddens, n_tasks, active_func='ReLU'):
        super(MLP, self).__init__()
        if active_func == 'ReLU':
            self.activation = nn.ReLU()
        elif active_func == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif active_func == 'GELU':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {active_func}")

        self.model = nn.Sequential(
            nn.Linear(n_feats, n_hiddens),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(n_hiddens, n_hiddens),
            self.activation,
            nn.Dropout(0.1),
            nn.Linear(n_hiddens, n_tasks),
        )
        self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x.to(torch.float32))
        return x