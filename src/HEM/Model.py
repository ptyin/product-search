import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, word_num, entity_num, embedding_size, l2):
        super().__init__()
        self.l2 = l2

        self.word_embedding_layer = nn.Embedding(word_num, embedding_size, padding_idx=0)
        self.log_sigmoid = nn.LogSigmoid()
        self.word_bias = nn.Embedding(word_num, 1, padding_idx=0)
        self.entity_embedding_layer = nn.Embedding(entity_num, embedding_size)
        # self.entity_bias = nn.Embedding(entity_num, 1)
        self.query_projection = nn.Linear(embedding_size, embedding_size)

        self.personalized_factor = nn.Parameter(torch.tensor([0.0]))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.word_embedding_layer.weight, 0, 0.1)
        with torch.no_grad():
            self.word_embedding_layer.weight[self.word_embedding_layer.padding_idx].fill_(0)
        nn.init.zeros_(self.word_bias.weight)
        with torch.no_grad():
            self.word_bias.weight[self.word_embedding_layer.padding_idx].fill_(0)
        # nn.init.normal_(self.entity_embedding_layer.weight, 0, 0.1)
        nn.init.zeros_(self.entity_embedding_layer.weight)

        nn.init.uniform_(self.personalized_factor)

        nn.init.xavier_normal_(self.query_projection.weight)
        nn.init.uniform_(self.query_projection.bias, 0, 0.1)

    def nce_loss(self, words, neg_words, entities):
        word_embeddings = self.word_embedding_layer(words)
        # (batch, embedding_size)
        word_biases = self.word_bias(words).squeeze(dim=1)
        # (batch, )
        neg_word_embeddings = self.word_embedding_layer(neg_words)
        # (batch, k, embedding_size)
        neg_word_biases = self.word_bias(neg_words).squeeze(dim=2)
        # (batch, k, )

        entity_embeddings = self.entity_embedding_layer(entities)
        # (batch, embedding_size)

        pos = -self.log_sigmoid(torch.sum(word_embeddings * entity_embeddings) + word_biases)
        neg = self.log_sigmoid(-torch.einsum('bke,be->bk', neg_word_embeddings, entity_embeddings) - neg_word_biases)
        neg = -torch.sum(neg, dim=1)
        return pos + neg

    def search_loss(self, user_embeddings, query_embeddings, item_embeddings, neg_item_embeddings):
        personalized_search_model = self.personalized_factor * query_embeddings +\
                                    (1 - self.personalized_factor) * user_embeddings
        # (batch, embedding_size)
        pos = -self.log_sigmoid(torch.sum(item_embeddings * personalized_search_model))
        neg = self.log_sigmoid(-torch.einsum('bke,be->bk', neg_item_embeddings, personalized_search_model))
        neg = -torch.sum(neg)
        return pos + neg

    def regularization_loss(self):
        return self.l2 * (self.word_embedding_layer.weight.norm() + self.entity_embedding_layer.weight.norm())

    def forward(self, users, items, query_words,
                mode: str,
                review_words=None,
                neg_items=None, neg_review_words=None):
        """
        Parameters
        -----
        users
            (batch, )
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
            item_embeddings = self.entity_embedding_layer(items)
            return item_embeddings
        user_embeddings = self.entity_embedding_layer(users)
        item_embeddings = self.entity_embedding_layer(items)
        query_embeddings = torch.mean(self.word_embedding_layer(query_words), dim=1)
        query_embeddings = torch.tanh(self.query_projection(query_embeddings))
        if mode == 'test':
            personalized_model = self.personalized_factor * query_embeddings +\
                                 (1 - self.personalized_factor) * user_embeddings

            return personalized_model, item_embeddings

        if mode == 'train':
            neg_item_embeddings = self.entity_embedding_layer(neg_items)
            search_loss = self.search_loss(user_embeddings, query_embeddings, item_embeddings, neg_item_embeddings)

            item_word_loss = self.nce_loss(review_words, neg_review_words, items)
            regularization_loss = self.regularization_loss()
            return (item_word_loss + search_loss).mean(dim=0) + regularization_loss
        else:
            raise NotImplementedError
