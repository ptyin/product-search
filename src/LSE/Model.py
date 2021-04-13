import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, word_num, item_num, embedding_size, l2):
        super(Model, self).__init__()

        self.l2 = l2
        self.word_embedding_layer = nn.Embedding(word_num, embedding_size, padding_idx=0)
        self.log_sigmoid = nn.LogSigmoid()
        # self.word_bias = nn.Embedding(word_num, 1, padding_idx=0)
        self.item_embedding_layer = nn.Embedding(item_num, embedding_size)
        self.item_bias = nn.Embedding(item_num, 1)
        self.query_projection = nn.Linear(embedding_size, embedding_size)

        self.personalized_factor = nn.Parameter(torch.tensor([0.0]))

    def reset_parameters(self):
        nn.init.normal_(self.word_embedding_layer.weight, 0, 0.1)
        with torch.no_grad():
            self.word_embedding_layer.weight[self.word_embedding_layer.padding_idx].fill_(0)

    def nce_loss(self, words, items, neg_items):
        word_embeddings = self.__fs(words)
        # (batch, embedding_size)

        item_embeddings = self.item_embedding_layer(items)
        neg_item_embeddings = self.item_embedding_layer(neg_items)
        # (batch, embedding_size)
        item_biases = self.item_bias(items).squeeze(dim=-1)
        neg_item_biases = self.item_bias(neg_items).squeeze(dim=-1)
        # (batch, embedding_size)

        pos = -self.log_sigmoid(torch.sum(word_embeddings * item_embeddings) + item_biases)
        neg = self.log_sigmoid(-torch.einsum('be,bke->bk', word_embeddings, neg_item_embeddings) - neg_item_biases)
        neg = -torch.sum(neg, dim=1)
        return pos + neg

    def __fs(self, words):
        embeddings = torch.mean(self.word_embedding_layer(words), dim=1)
        embeddings = torch.tanh(self.query_projection(embeddings))
        return embeddings

    def regularization_loss(self):
        return self.l2 * (self.word_embedding_layer.weight.norm() + self.item_embedding_layer.weight.norm())

    def forward(self, items, query_words,
                mode: str,
                review_words=None,
                neg_items=None):
        """
        Parameters
        -----
        items
            (batch, )
        query_words
            (batch, word_num)
        mode
        review_words
            (batch, )
        neg_items
            (batch, k)
        """
        if mode == 'output_embedding':
            item_embeddings = self.item_embedding_layer(items)
            return item_embeddings
        item_embeddings = self.item_embedding_layer(items)
        query_embeddings = self.__fs(query_words)

        if mode == 'test':
            return query_embeddings, item_embeddings

        if mode == 'train':
            item_word_loss = self.nce_loss(review_words, items, neg_items)
            regularization_loss = self.regularization_loss()
            loss = item_word_loss.mean(dim=0) + regularization_loss
            return loss
        else:
            raise NotImplementedError
