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
        self.word_embedding_layer = nn.Embedding(word_num, embedding_size, padding_idx=0)
        self.log_sigmoid = nn.LogSigmoid()
        self.word_bias = nn.Embedding(word_num, 1, padding_idx=0)

        self.item_embedding_layer = nn.Embedding(item_num + 1, embedding_size, padding_idx=0)
        self.attention_layer = AttentionLayer(embedding_size, attention_hidden_dim, self.__class__.__name__)

        self.query_projection = nn.Linear(embedding_size, embedding_size)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.word_embedding_layer.weight, 0, 0.1)
        with torch.no_grad():
            self.word_embedding_layer.weight[self.word_embedding_layer.padding_idx].fill_(0)
        nn.init.zeros_(self.word_bias.weight)
        with torch.no_grad():
            self.word_bias.weight[self.word_embedding_layer.padding_idx].fill_(0)

        nn.init.zeros_(self.item_embedding_layer.weight)
        nn.init.xavier_normal_(self.query_projection.weight)
        nn.init.uniform_(self.query_projection.bias, 0, 0.1)
        self.attention_layer.reset_parameters()

    def nce_loss(self, words, neg_words, item_embeddings):
        word_embeddings = self.word_embedding_layer(words)
        # (batch, embedding_size)
        word_biases = self.word_bias(words).squeeze(dim=1)
        # (batch, )
        neg_word_embeddings = self.word_embedding_layer(neg_words)
        # (batch, k, embedding_size)
        neg_word_biases = self.word_bias(neg_words).squeeze(dim=2)
        # (batch, k, )

        pos = -self.log_sigmoid(torch.sum(word_embeddings * item_embeddings) + word_biases)
        neg = self.log_sigmoid(
            -torch.einsum('bke,be->bk', neg_word_embeddings, item_embeddings) - neg_word_biases)
        neg = -torch.sum(neg, dim=1)
        return pos + neg

    def search_loss(self, user_embeddings, query_embeddings, item_embeddings, neg_item_embeddings):
        personalized_search_model = query_embeddings + user_embeddings
        # (batch, embedding_size)
        pos = -self.log_sigmoid(torch.sum(item_embeddings * personalized_search_model))
        neg = self.log_sigmoid(-torch.einsum('bke,be->bk', neg_item_embeddings, personalized_search_model))
        neg = -torch.sum(neg)
        return pos + neg

    # def regularization_loss(self):
    #     return self.l2 * (self.word_embedding_layer.weight.norm() + self.item_embedding_layer.weight.norm())

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
        item_embeddings = self.item_embedding_layer(items)
        if mode == 'output_embedding':
            return item_embeddings
        user_bought_embeddings = self.item_embedding_layer(user_bought_items)
        query_embeddings = torch.mean(self.word_embedding_layer(query_words), dim=1)
        query_embeddings = torch.tanh(self.query_projection(query_embeddings))
        user_embeddings = self.attention_layer(user_bought_embeddings, query_embeddings)

        if mode == 'test':
            personalized_model = query_embeddings + user_embeddings
            return personalized_model, item_embeddings
        elif mode == 'train':
            neg_item_embeddings = self.item_embedding_layer(neg_items)
            search_loss = self.search_loss(user_embeddings, query_embeddings, item_embeddings, neg_item_embeddings)

            item_word_loss = self.nce_loss(review_words, neg_review_words, item_embeddings)
            # regularization_loss = self.regularization_loss()
            regularization_loss = 0
            return (item_word_loss + search_loss).mean(dim=0) + regularization_loss
        else:
            raise NotImplementedError


class ZAM(AEM):
    def __init__(self, word_num, item_num, embedding_size, attention_hidden_dim):
        super().__init__(word_num, item_num, embedding_size, attention_hidden_dim)
        # self.attention_layer = AttentionLayer(embedding_size, attention_hidden_dim, 'ZAM')
