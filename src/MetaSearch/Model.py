import torch
from torch import nn
from MetaSearch.self_attention.MultiHeadSelfAttention import MultiHeadSelfAttention
from MetaSearch.self_attention.Mean import Mean
from MetaSearch.experiment.loss import *
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.query_projection = nn.Linear(input_dim, input_dim * hidden_dim)
        self.reduce_projection = nn.Linear(hidden_dim, 1, bias=False)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.query_projection.weight)
        nn.init.uniform_(self.query_projection.bias, 0, 0.01)
        nn.init.xavier_normal_(self.reduce_projection.weight)

    def forward(self, reviews_embedding, query_embedding):
        """
        Parameters
        -----------
        reviews_embedding: shape(batch, review_num, input_dim)
        query_embedding: shape(batch, 1, input_dim) or (batch, input_dim)

       Return
       -----------
       torch.Tensor: shape(batch, input_dim)
        """
        batch_size = len(query_embedding)
        # ------------tanh(W*q+b)------------
        projected_query = torch.tanh(self.query_projection(query_embedding))
        # shape: (batch, 1, input_dim * hidden_dim) or (batch, input_dim * hidden_dim)
        projected_query = projected_query.view((batch_size, self.input_dim, self.hidden_dim))
        # shape: (batch, input_dim, hidden_dim)
        # ------------r@tanh(W*q+b)------------
        reviews_query_dotted_sum = torch.einsum('bri,bih->brh', reviews_embedding, projected_query)
        # shape: (batch, review_num, hidden_dim)
        # ------------(r*tanh(W_1*q+b))*W_2------------
        reviews_query_reduce_sum = self.reduce_projection(reviews_query_dotted_sum)
        # shape: (batch, review_num, 1)
        weight = torch.softmax(reviews_query_reduce_sum, dim=1)

        entity_embedding = torch.sum(weight * reviews_embedding, dim=1)
        # shape: (batch, input_dim)
        return entity_embedding


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.reduce_projection = nn.Linear(input_dim, 1, bias=False)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.reduce_projection.weight)
        # nn.init.uniform_(self.reduce_projection.bias, 0, 0.01)

    def forward(self, words_embedding: torch.Tensor):
        """
        Parameters
        -----------
        words_embedding
            shape(batch, review_num, word_num, input_dim)
            or   (batch, 1, word_num, input_dim) for query

        Return
        -----------
        torch.Tensor
            shape(batch, review_num, input_dim)
        """
        original_shape = words_embedding.shape
        # ------------tanh(W*w+b)------------
        reduced_words = torch.tanh(self.reduce_projection(words_embedding))
        # shape: (batch, review_num, word_num, 1)
        weight = torch.softmax(reduced_words, dim=2)
        # shape: (batch, review_num, word_num, 1)
        entity_embedding = torch.sum(weight * words_embedding, dim=2)
        # shape: (batch, review_num, input_dim)
        return entity_embedding


class Model(nn.Module):
    def __init__(self, word_num, user_num, item_num,
                 embedding_size, attention_hidden_dim, head_num):
        super(Model, self).__init__()
        self.word_num = word_num
        self.user_num = user_num
        self.item_num = item_num

        self.embedding_size = embedding_size
        self.attention_hidden_dim = attention_hidden_dim

        self.word_embedding_layer = nn.Embedding(self.word_num, self.embedding_size, padding_idx=0)
        # self.user_embedding_layer = nn.Embedding(self.user_num, self.embedding_size)
        # self.item_embedding_layer = nn.Embedding(self.item_num, self.embedding_size)
        # self.entity_embedding_layer = nn.Embedding(self.entity_num, self.embedding_size)
        self.doc_embedding_layer = SelfAttentionLayer(self.embedding_size)
        # self.doc_embedding_layer = nn.Sequential(MultiHeadSelfAttention(self.embedding_size,
        #                                                                 self.embedding_size // head_num,
        #                                                                 head_num),
        #                                          Mean(dim=-2))

        self.attention_layer = AttentionLayer(self.embedding_size, self.attention_hidden_dim)
        # self.personalized_factor = nn.Parameter(torch.tensor(0.5))
        self.personalized_factor = 1

        # self.global_parameters: list = [self.word_embedding_layer.weight,
        #                                 self.doc_embedding_layer.reduce_projection.weight]
        # self.global_parameters: list = [self.word_embedding_layer.weight,
        #                                 self.user_embedding_layer.weight,
        #                                 self.item_embedding_layer.weight]
        self.global_parameters: list = [self.word_embedding_layer.weight]
        self.reset_parameters()

    def reset_parameters(self):
        # self.word_embedding_layer.reset_parameters()
        nn.init.uniform_(self.word_embedding_layer.weight, 0, 0.1)
        with torch.no_grad():
            self.word_embedding_layer.weight[self.word_embedding_layer.padding_idx].fill_(0)
        # nn.init.uniform_(self.user_embedding_layer.weight, 0, 0.1)
        # nn.init.uniform_(self.item_embedding_layer.weight, 0, 0.1)
        self.doc_embedding_layer.reset_parameters()
        # for layer in self.doc_embedding_layer:
        #     layer.reset_parameters()
        self.attention_layer.reset_parameters()
        # nn.init.uniform_(self.personalized_factor)

    def set_local(self):
        for global_parameters in self.global_parameters:
            global_parameters.requires_grad = False

    def set_global(self):
        for global_parameters in self.global_parameters:
            global_parameters.requires_grad = True

    def embedding(self, words):
        word_embedding = self.word_embedding_layer(words)
        doc_embedding = self.doc_embedding_layer(word_embedding)
        return doc_embedding

    def regularization_term(self):
        norm = 0
        for para in self.parameters():
            norm += para.norm()
        return norm

    def forward(self,
                users, user_reviews_words,
                items, item_reviews_words,
                query,
                mode,
                negative_items=None, negative_item_reviews_words=None):
        """
        Parameters
        -----
        users: torch.LongTensor
            shape: (batch,)
        user_reviews_words: torch.LongTensor
            shape: (batch, review_num, word_num)
        items: torch.LongTensor
            shape: (batch,)
        item_reviews_words: torch.LongTensor
            shape: (batch, review_num, word_num)
        query: torch.LongTensor
            shape: (batch, word_num)
        mode: str
            'train', 'test', 'output_embedding'
        negative_items: torch.LongTensor
            shape: (batch,)
        negative_item_reviews_words: torch.LongTensor
            shape: (batch, review_num, word_num)
        """
        if mode == 'output_embedding':
            # item_embeddings = self.item_embedding_layer(items)
            # return item_embeddings

            item_reviews_embedding = self.embedding(item_reviews_words)
            # query_embedding = self.embedding(query.unsqueeze(dim=1))
            # item_entity = self.attention_layer(item_reviews_embedding, query_embedding).squeeze(dim=0)
            item_entity = torch.mean(item_reviews_embedding, dim=-2)
            return item_entity

        # user_embeddings = self.user_embedding_layer(users)
        query_embedding = self.embedding(query.unsqueeze(dim=1))
        query_embedding = query_embedding.squeeze(dim=1)

        # personalized_model = user_embeddings + query_embedding
        user_reviews_embedding = self.embedding(user_reviews_words)
        user_entity = self.attention_layer(user_reviews_embedding, query_embedding)
        # user_entity = torch.mean(user_reviews_embedding, dim=-2)
        personalized_model = user_entity + query_embedding

        if mode == 'train':
            # user_reviews_embedding = self.embedding(user_reviews_words)
            # user_entity = self.attention_layer(user_reviews_embedding, query_embedding)

            # item_embeddings = self.item_embedding_layer(items)
            item_reviews_embedding = self.embedding(item_reviews_words)
            # item_entity = self.attention_layer(item_reviews_embedding, query_embedding)
            item_entity = torch.mean(item_reviews_embedding, dim=-2)
            # item_entity = torch.mean(item_reviews_embedding, dim=-2)

            # negative_item_embeddings = self.item_embedding_layer(negative_items)
            negative_item_reviews_embedding = self.embedding(negative_item_reviews_words)
            # negative_item_entity = self.attention_layer(negative_item_reviews_embedding, query_embedding)
            negative_item_entity = torch.mean(negative_item_reviews_embedding, dim=-2)

            # search_loss = triplet_margin_loss(personalized_model, item_entity, negative_item_entity)
            search_loss = hem_loss(personalized_model, item_entity, negative_item_entity)
            # search_loss = hem_loss(personalized_model, item_embeddings, negative_item_embeddings)
            # user_loss = log_loss(user_embeddings, user_entity)
            # item_loss = log_loss(item_embeddings, item_entity)

            # return search_loss + user_loss + item_loss
            return search_loss
        elif mode == 'test':
            return personalized_model
        else:
            raise NotImplementedError
