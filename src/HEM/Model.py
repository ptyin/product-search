import torch
from torch import nn
import torch.nn.functional as function


class Model(nn.Module):
    def __init__(self, word_num, entity_num, embedding_size, l2):
        super().__init__()
        self.embedding_size = embedding_size
        self.l2 = l2

        self.word_embedding_layer = nn.Embedding(word_num, embedding_size, padding_idx=word_num-1)
        # self.log_sigmoid = nn.LogSigmoid()
        # self.cross_entropy = nn.BCEWithLogitsLoss()
        self.word_bias = nn.Embedding(word_num, 1, padding_idx=word_num-1)
        self.entity_embedding_layer = nn.Embedding(entity_num, embedding_size)
        self.entity_bias = nn.Embedding(entity_num, 1)
        self.query_projection = nn.Linear(embedding_size, embedding_size, bias=True)

        # self.personalized_factor = nn.Parameter(torch.tensor([0.5]))
        self.personalized_factor = 0.5

        self.reset_parameters()

    def reset_parameters(self):
        init_width = 0.5 / self.embedding_size
        # nn.init.normal_(self.word_embedding_layer.weight, 0, 0.1)
        nn.init.uniform_(self.word_embedding_layer.weight, -init_width, init_width)
        with torch.no_grad():
            self.word_embedding_layer.weight[self.word_embedding_layer.padding_idx].fill_(0)
        nn.init.zeros_(self.word_bias.weight)
        with torch.no_grad():
            self.word_bias.weight[self.word_bias.padding_idx].fill_(0)
        # nn.init.normal_(self.entity_embedding_layer.weight, 0, 0.1)
        nn.init.zeros_(self.entity_embedding_layer.weight)
        nn.init.zeros_(self.entity_bias.weight)

        # nn.init.xavier_normal_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.query_projection.weight)
        # nn.init.uniform_(self.query_projection.bias, 0, 0.01)

    def nce_loss(self, words, neg_words, entity_embeddings):
        word_embeddings = self.word_embedding_layer(words)
        # (batch, embedding_size)
        word_biases = self.word_bias(words).squeeze(dim=1)
        # (batch, )
        neg_word_embeddings: torch.Tensor = self.word_embedding_layer(neg_words)
        # (k, embedding_size)
        neg_word_biases = self.word_bias(neg_words).squeeze(dim=1)
        # (k, )

        # pos = -self.log_sigmoid(torch.sum(word_embeddings * entity_embeddings, dim=1) + word_biases)
        # neg = self.log_sigmoid(-torch.einsum('bke,be->bk', neg_word_embeddings, entity_embeddings) - neg_word_biases)
        # neg = self.log_sigmoid(-entity_embeddings @ neg_word_embeddings.t() - neg_word_biases)
        # neg = torch.log(1 - torch.sigmoid(entity_embeddings @ neg_word_embeddings.t() + neg_word_biases))
        pos = torch.sum(word_embeddings * entity_embeddings, dim=1) + word_biases
        pos = function.binary_cross_entropy_with_logits(pos, torch.ones_like(pos).cuda(), reduction='none')
        neg = entity_embeddings @ neg_word_embeddings.t() + neg_word_biases
        neg = function.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg).cuda(), reduction='none')
        neg = torch.sum(neg, dim=1)
        return pos + neg

    def search_loss(self, user_embeddings, query_embeddings, item_embeddings, neg_item_embeddings,
                    item_biases, neg_biases):
        personalized_search_model: torch.Tensor = self.personalized_factor * query_embeddings +\
                                                  (1 - self.personalized_factor) * user_embeddings
        # (batch, embedding_size)
        # pos = -self.log_sigmoid(torch.sum(item_embeddings * personalized_search_model, dim=1) + item_biases)
        # neg = self.log_sigmoid(-torch.einsum('bke,be->bk', neg_item_embeddings, personalized_search_model) - neg_biases)
        # neg = self.log_sigmoid(-personalized_search_model @ neg_item_embeddings.t() - neg_biases)
        # neg = torch.log(1 - torch.sigmoid(personalized_search_model @ neg_item_embeddings.t() + neg_biases))
        pos = torch.sum(item_embeddings * personalized_search_model, dim=1) + item_biases
        pos = function.binary_cross_entropy_with_logits(pos, torch.ones_like(pos).cuda(), reduction='none')
        neg = personalized_search_model @ neg_item_embeddings.t() + neg_biases
        neg = function.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg).cuda(), reduction='none')
        neg = torch.sum(neg, dim=1)
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
            (k, )
        neg_review_words
            (k, )
        """
        if mode == 'output_embedding':
            item_embeddings = self.entity_embedding_layer(items)
            item_biases = self.entity_bias(items).squeeze(dim=1)
            # return item_embeddings
            return item_embeddings, item_biases
        user_embeddings = self.entity_embedding_layer(users)
        # item_embeddings = self.entity_embedding_layer(items)
        query_embeddings = torch.mean(self.word_embedding_layer(query_words), dim=1)
        query_embeddings = torch.tanh(self.query_projection(query_embeddings))
        if mode == 'test':
            personalized_model = self.personalized_factor * query_embeddings +\
                                 (1 - self.personalized_factor) * user_embeddings

            # return personalized_model, item_embeddings
            return personalized_model

        if mode == 'train':
            item_embeddings = self.entity_embedding_layer(items)
            neg_item_embeddings = self.entity_embedding_layer(neg_items)

            user_word_loss = self.nce_loss(review_words, neg_review_words, user_embeddings).mean(dim=0)
            item_word_loss = self.nce_loss(review_words, neg_review_words, item_embeddings).mean(dim=0)
            item_biases, neg_biases = self.entity_bias(items).squeeze(dim=1), self.entity_bias(neg_items).squeeze(dim=1)
            search_loss = self.search_loss(user_embeddings, query_embeddings, item_embeddings, neg_item_embeddings,
                                           item_biases, neg_biases).mean(dim=0)

            regularization_loss = self.regularization_loss()
            return (user_word_loss + item_word_loss + search_loss) + regularization_loss
        else:
            raise NotImplementedError
