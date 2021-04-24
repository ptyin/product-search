import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, head_num):
        super(MultiHeadSelfAttention, self).__init__()
        self.factor = math.sqrt(input_dim)
        self.head_num = head_num

        self.weight_q = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for i in range(head_num)])
        self.weight_k = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for i in range(head_num)])
        self.weight_v = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False) for i in range(head_num)])

        self.weight_o = nn.Linear(hidden_dim * head_num, input_dim, bias=False)

    def reset_parameters(self):
        for module_list in [self.weight_q, self.weight_k, self.weight_v]:
            for layer in module_list:
                nn.init.xavier_normal_(layer.weight)

    def __attention_score(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        :param q: (batch, seq, dim)
        :param k: (batch, seq, dim)
        :param v: (batch, seq, dim)
        :return: (batch, seq, dim)
        """
        dot_product = q @ k.transpose(1, 2)
        scaled_dot_product = dot_product / self.factor
        weight = torch.softmax(scaled_dot_product, dim=2)
        # shape: (batch, seq, seq)
        score = weight @ v
        return score

    def forward(self, words_embeddings):
        """
        :param words_embeddings: (batch, seq, dim)
        :return: (batch, seq, dim)
        """

        z = []
        for i in range(self.head_num):
            q = self.weight_q[i](words_embeddings)
            k = self.weight_k[i](words_embeddings)
            v = self.weight_v[i](words_embeddings)
            z.append(self.__attention_score(q, k, v))
        concatenated_z = torch.cat(z, dim=2)
        output = self.weight_o(concatenated_z)
        return output
