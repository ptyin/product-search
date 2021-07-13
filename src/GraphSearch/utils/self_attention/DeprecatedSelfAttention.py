import torch
from torch import nn


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.reduce_projection = nn.Linear(input_dim, 1)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.reduce_projection.weight)
        nn.init.uniform_(self.reduce_projection.bias, 0, 0.01)

    def forward(self, words_embedding: torch.Tensor):
        """
        Parameters
        -----------
        words_embedding: torch.Tensor
            shape(batch, sequence, input_dim)

        Return
        -----------
        torch.Tensor
            shape(batch, input_dim)
        """
        # TODO deprecated, change to multi-head self attention
        original_shape = words_embedding.shape
        # ------------tanh(W*w+b)------------
        reduced_words = torch.tanh(self.reduce_projection(words_embedding))
        # shape: (batch, sequence, 1)
        weight = torch.softmax(reduced_words, dim=1)
        # shape: (batch, sequence, 1)
        entity_embedding = torch.sum(weight * words_embedding, dim=1)
        # shape: (batch, input_dim)
        return entity_embedding

