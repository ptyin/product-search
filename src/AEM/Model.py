import torch
from torch import nn
from common.loss import nce_loss
import math


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, head_num, model_name: str):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.head_num = head_num
        self.model_name = model_name

        self.query_projection = nn.Linear(input_dim, input_dim * head_num)
        self.reduce_projection = nn.Linear(head_num, 1, bias=False)

        # self.dropout = nn.Dropout()

    def reset_parameters(self):

        # nn.init.xavier_uniform_(self.query_projection.weight)

        fan_in, fan_out = self.input_dim, self.input_dim
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        nn.init.uniform_(self.query_projection.weight, -a, a)

        fan_in, fan_out = self.input_dim, 1
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        nn.init.uniform_(self.query_projection.bias, -a, a)

        nn.init.uniform_(self.reduce_projection.weight, math.sqrt(3.0), math.sqrt(3.0))
        # nn.init.xavier_uniform_(self.reduce_projection.weight)

    def attention_function(self, query_embedding, item_embedding):
        batch_size = len(query_embedding)

        # ------------tanh(W*q+b)------------
        projected_query = torch.tanh(self.query_projection(query_embedding))
        # projected_query = self.dropout(projected_query)
        # shape: (batch, 1, input_dim * hidden_dim) or (batch, input_dim * hidden_dim)
        projected_query = projected_query.view((batch_size, self.input_dim, self.head_num))
        # shape: (batch, input_dim, hidden_dim)

        # ------------r*tanh(W*q+b)------------
        # items_query_dotted_sum = torch.einsum('bri,bih->brh', item_embedding, projected_query)
        items_query_dotted_sum = item_embedding @ projected_query
        # shape: (batch, bought_item_num, hidden_dim)
        # ------------(r*tanh(W_1*q+b))*W_2------------
        items_query_reduce_sum = self.reduce_projection(items_query_dotted_sum)
        # shape: (batch, bought_item_num, 1)

        # scores = items_query_reduce_sum
        # the line below is copied from the original source code yet inconsistent with the paper
        scores = (items_query_reduce_sum - torch.max(items_query_reduce_sum, dim=1, keepdim=True)[0])

        return scores

    def forward(self, item_embedding, query_embedding, mask):
        """
        Parameters
        -----------
        item_embedding: shape(batch, bought_item_num, input_dim)
        query_embedding: shape(batch, 1, input_dim) or (batch, input_dim)
        mask: shape(batch, bought_item_num, )

       Return
       -----------
       torch.Tensor: shape(batch, input_dim)
        """
        if self.model_name == 'AEM':
            attention_score = self.attention_function(query_embedding, item_embedding)
            # shape: (batch, bought_item_num, 1)
            if mask is not None:
                # attention_score = attention_score.masked_fill(mask, -float('inf'))
                attention_score = torch.exp(attention_score) * mask
            # weight = torch.softmax(attention_score, dim=1)
            denominator = torch.sum(attention_score, dim=1, keepdim=True)
            weight = attention_score / torch.where(torch.less(denominator, 1e-7), denominator + 1, denominator)
        elif self.model_name == 'ZAM':
            item_embedding = torch.cat([torch.zeros(item_embedding.shape[0], 1, self.input_dim, device='cuda:0'),
                                        item_embedding], dim=1)
            attention_score = self.attention_function(query_embedding, item_embedding)
            if mask is not None:
                mask = torch.cat([torch.ones(item_embedding.shape[0], 1, 1, dtype=torch.bool, device='cuda:0'),
                                  mask], dim=1)
                # attention_score = attention_score.masked_fill(mask, -float('inf'))
                attention_score = torch.exp(attention_score) * mask
            # weight = torch.softmax(attention_score, dim=1)
            denominator = torch.sum(attention_score, dim=1, keepdim=True)
            weight = attention_score / torch.where(torch.less(denominator, 1e-7), denominator + 1, denominator)
        else:
            raise NotImplementedError
        # shape: (batch, bought_item_num, 1)

        entity_embedding = torch.sum(weight * item_embedding, dim=1)
        # shape: (batch, input_dim)
        return entity_embedding


class AEM(nn.Module):
    def __init__(self, word_num, item_num, embedding_size, attention_hidden_dim, l2):
        super().__init__()
        self.embedding_size = embedding_size
        self.l2 = l2

        self.word_embedding_layer = nn.Embedding(word_num, embedding_size, padding_idx=0)
        # self.log_sigmoid = nn.LogSigmoid()
        self.word_bias = nn.Embedding(word_num, 1, padding_idx=0)

        self.item_embedding_layer = nn.Embedding(item_num, embedding_size, padding_idx=0)
        self.item_bias = nn.Embedding(item_num, 1, padding_idx=0)
        self.attention_layer = AttentionLayer(embedding_size, attention_hidden_dim, self.__class__.__name__)

        self.query_projection = nn.Linear(embedding_size, embedding_size)

        self.reset_parameters()

    def reset_parameters(self):
        init_width = 0.5 / self.embedding_size
        # nn.init.normal_(self.word_embedding_layer.weight, 0, 0.1)
        nn.init.uniform_(self.word_embedding_layer.weight, -init_width, init_width)
        with torch.no_grad():
            self.word_embedding_layer.weight[self.word_embedding_layer.padding_idx].fill_(0)
        nn.init.zeros_(self.word_bias.weight)
        with torch.no_grad():
            self.word_bias.weight[self.word_embedding_layer.padding_idx].fill_(0)

        nn.init.zeros_(self.item_embedding_layer.weight)
        with torch.no_grad():
            self.item_embedding_layer.weight[self.item_embedding_layer.padding_idx].fill_(0)
        nn.init.zeros_(self.item_bias.weight)
        with torch.no_grad():
            self.item_bias.weight[self.item_bias.padding_idx].fill_(0)

        # nn.init.xavier_normal_(self.query_projection.weight)
        nn.init.xavier_uniform_(self.query_projection.weight)

        fan_in, fan_out = self.embedding_size, 1
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        nn.init.uniform_(self.query_projection.bias, -a, a)

        self.attention_layer.reset_parameters()

    def regularization_loss(self):
        return self.l2 * (self.word_embedding_layer.weight.norm() +
                          self.item_embedding_layer.weight.norm() +
                          self.query_projection.weight.norm() +
                          self.attention_layer.query_projection.weight.norm() +
                          self.attention_layer.reduce_projection.weight.norm())

    def forward(self, user_bought_items, items, query_words,
                mode: str,
                user_bought_masks=None,
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
        user_bought_masks
            (batch, bought_items)
        review_words
            (batch, )
        neg_items
            (batch, k)
        neg_review_words
            (batch, k)
        """
        if mode == 'output_embedding':
            item_embeddings = self.item_embedding_layer(items)
            item_biases = self.item_bias(items).squeeze(dim=1)

            return item_embeddings, item_biases
        user_bought_embeddings = self.item_embedding_layer(user_bought_items)
        query_embeddings = torch.mean(self.word_embedding_layer(query_words), dim=1)
        query_embeddings = torch.tanh(self.query_projection(query_embeddings))
        user_embeddings = self.attention_layer(user_bought_embeddings, query_embeddings, mask=user_bought_masks)
        # user_embeddings = user_bought_embeddings.mean(dim=1)
        # user_embeddings = user_bought_embeddings.sum(dim=1) / (~user_bought_masks).sum(dim=1, keepdim=True)

        personalized_model = 0.5 * (query_embeddings + user_embeddings)
        # personalized_model = query_embeddings

        if mode == 'test':
            return personalized_model
        elif mode == 'train':
            item_embeddings = self.item_embedding_layer(items)
            neg_item_embeddings = self.item_embedding_layer(neg_items)
            word_embeddings = self.word_embedding_layer(review_words)
            # (batch, embedding_size)
            word_biases = self.word_bias(review_words).squeeze(dim=1)
            # (batch, )
            neg_word_embeddings = self.word_embedding_layer(neg_review_words)
            # (k, embedding_size)
            neg_word_biases = self.word_bias(neg_review_words).squeeze(dim=1)
            # (k, )

            item_word_loss = nce_loss(item_embeddings,
                                      word_embeddings, neg_word_embeddings,
                                      word_biases, neg_word_biases).mean(dim=0)
            item_biases, neg_biases = self.item_bias(items).squeeze(dim=1), self.item_bias(neg_items).squeeze(dim=1)
            search_loss = nce_loss(personalized_model,
                                   item_embeddings, neg_item_embeddings,
                                   item_biases, neg_biases).mean(dim=0)

            regularization_loss = self.regularization_loss()
            return item_word_loss + search_loss + regularization_loss
            # return item_word_loss
        else:
            raise NotImplementedError


class ZAM(AEM):
    def __init__(self, word_num, item_num, embedding_size, attention_hidden_dim, l2):
        super().__init__(word_num, item_num, embedding_size, attention_hidden_dim, l2)
        # self.attention_layer = AttentionLayer(embedding_size, attention_hidden_dim, 'ZAM')
