import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import learn2learn as l2l


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
        self.reduce_projection = nn.Linear(input_dim, 1)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.reduce_projection.weight)
        nn.init.uniform_(self.reduce_projection.bias, 0, 0.01)

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
    def __init__(self, word_num, word_embedding_size,
                 attention_hidden_dim):
        super(Model, self).__init__()
        self.word_num = word_num
        self.word_embedding_size = word_embedding_size
        self.attention_hidden_dim = attention_hidden_dim

        self.word_embedding_layer = nn.Embedding(self.word_num, self.word_embedding_size, padding_idx=0)
        self.doc_embedding_layer = SelfAttentionLayer(self.word_embedding_size)
        self.attention_layer = AttentionLayer(self.word_embedding_size, self.attention_hidden_dim)
        self.personalized_factor = nn.Parameter(torch.tensor(0.0))

        self.global_parameters: list = [self.word_embedding_layer.weight, *self.doc_embedding_layer.parameters()]

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.word_embedding_layer.weight, 0.0, 0.01)
        self.doc_embedding_layer.reset_parameters()
        self.attention_layer.reset_parameters()
        nn.init.uniform_(self.personalized_factor)

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
                user_reviews_words: torch.LongTensor, item_reviews_words: torch.LongTensor,
                query: torch.LongTensor,
                mode,
                negative_item_reviews_words: torch.LongTensor = None):
        """
        Parameters
        -----
        user_reviews_words
            shape: (batch, review_num, word_num)
        item_reviews_words
            shape: (batch, review_num, word_num)
        query
            shape: (batch, word_num)
        mode: str
            'train', 'test', 'output_embedding'
        negative_item_reviews_words
            shape: (batch, review_num, word_num)
        """
        if mode == 'output_embedding':
            item_reviews_embedding = self.embedding(item_reviews_words)
            query_embedding = self.embedding(query.unsqueeze(dim=1))
            item_entity = self.attention_layer(item_reviews_embedding, query_embedding).squeeze(dim=0)
            return item_entity

        user_reviews_embedding = self.embedding(user_reviews_words)
        item_reviews_embedding = self.embedding(item_reviews_words)
        query_embedding = self.embedding(query.unsqueeze(dim=1))

        user_entity = self.attention_layer(user_reviews_embedding, query_embedding)
        item_entity = self.attention_layer(item_reviews_embedding, query_embedding)

        query_embedding = query_embedding.squeeze(dim=1)
        personalized_model = user_entity + self.personalized_factor * query_embedding

        if mode == 'train':
            negative_item_reviews_embedding = self.embedding(negative_item_reviews_words)
            negative_item_entity = self.attention_layer(negative_item_reviews_embedding, query_embedding)

            return personalized_model, item_entity, negative_item_entity
        elif mode == 'test':
            return personalized_model, item_entity
