import math
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model:
        :param d_ff: feed_forward_hidden, usually 4*hidden_size
        :param dropout:
        """
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

        nn.init.uniform_(self.w_1.bias)
        nn.init.uniform_(self.w_2.bias)

    @staticmethod
    def activation(x):
        # GELU
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
