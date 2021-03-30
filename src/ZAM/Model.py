import torch
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, model_name: str):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model_name = model_name
        self.query_projection = nn.Linear(input_dim, input_dim * hidden_dim)
        self.reduce_projection = nn.Linear(hidden_dim, 1, bias=False)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.query_projection.weight)
        nn.init.uniform_(self.query_projection.bias, 0, 0.01)
        nn.init.xavier_normal_(self.reduce_projection.weight)

    def attention_function(self, query_embedding, item_embedding):
        batch_size = len(query_embedding)
        # ------------tanh(W*q+b)------------
        projected_query = torch.tanh(self.query_projection(query_embedding))
        # shape: (batch, 1, input_dim * hidden_dim) or (batch, input_dim * hidden_dim)
        projected_query = projected_query.view((batch_size, self.input_dim, self.hidden_dim))
        # shape: (batch, input_dim, hidden_dim)
        # ------------r@tanh(W*q+b)------------
        items_query_dotted_sum = torch.einsum('bri,bih->brh', item_embedding, projected_query)
        # shape: (batch, bought_item_num, hidden_dim)
        # ------------(r*tanh(W_1*q+b))*W_2------------
        items_query_reduce_sum = self.reduce_projection(items_query_dotted_sum)
        # shape: (batch, bought_item_num, 1)

        return items_query_reduce_sum

    def forward(self, item_embedding, query_embedding):
        """
        Parameters
        -----------
        item_embedding: shape(batch, bought_item_num, input_dim)
        query_embedding: shape(batch, 1, input_dim) or (batch, input_dim)

       Return
       -----------
       torch.Tensor: shape(batch, input_dim)
        """
        attention_score = self.attention_function(query_embedding, item_embedding)
        if self.model_name == 'AEM':
            weight = torch.softmax(attention_score, dim=1)
        elif self.model_name == 'ZAM':
            attention_score = torch.exp(attention_score)
            weight = attention_score / (1 + torch.sum(attention_score, dim=1))
        else:
            raise NotImplementedError
        # shape: (batch, bought_item_num, 1)

        entity_embedding = torch.sum(weight * item_embedding, dim=1)
        # shape: (batch, input_dim)
        return entity_embedding


class AEM(nn.Module):
    def __init__(self, word_num, item_num, embedding_size, attention_hidden_dim):
        super().__init__()
        self.word_embedding_layer = nn.Embedding(word_num, embedding_size)
        self.item_embedding_layer = nn.Embedding(item_num + 1, embedding_size, padding_idx=item_num)
        self.attention_layer = AttentionLayer(embedding_size, attention_hidden_dim, self.__name__)

    def reset_parameters(self):
        self.word_embedding_layer.reset_parameters()
        self.item_embedding_layer.reset_parameters()
        self.attention_layer.reset_parameters()

    def forward(self, user_bought_items, items, query_words,
                mode: str,
                review_words=None,
                neg_items=None, neg_review_words=None):
        """
        Parameters
        -----
        user_bought_items
            (batch, bought_items)
        items
            (batch, )
        query_words
            (batch, word_num)
        mode
        review_words
            (batch, )
        neg_items
            (batch, k)
        neg_review_words
            (batch, k)
        """
        if mode == 'output_embedding':
            item_embeddings = self.item_embedding_layer(items)


class ZAM(AEM):
    def __init__(self, word_num, item_num, embedding_size, attention_hidden_dim):
        super().__init__(word_num, item_num, embedding_size, attention_hidden_dim)
        # self.attention_layer = AttentionLayer(embedding_size, attention_hidden_dim, 'ZAM')
