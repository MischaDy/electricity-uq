from torch import nn
from more_itertools import collapse


class My_NN(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_hidden_layers=2,
        hidden_layer_size=50,
        activation=nn.LeakyReLU,
    ):
        super(My_NN, self).__init__()
        # fmt: off
        layers = collapse([
            nn.Linear(dim_in, hidden_layer_size),
            activation(),
            [[nn.Linear(hidden_layer_size, hidden_layer_size),
              activation()]
             for _ in range(num_hidden_layers)],
            nn.Linear(hidden_layer_size, dim_out),
        ])
        self.model = nn.Sequential(*layers).float()

    def forward(self, input_):
        # if len(input_) == 2:  # todo: fix properly!
        #     X, y = input_
        # else:
        #     X = input_
        X = input_
        return self.model(X)
