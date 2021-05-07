import torch
from torch import nn
import dgl
from dgl import udf
import dgl.function as fn

from GraphSearch.utils.self_attention.MultiHeadSelfAttention import MultiHeadSelfAttention
from GraphSearch.utils.self_attention.FeedForward import FeedForward
from GraphSearch.utils.self_attention.Mean import Mean
from GraphSearch.utils.experiment.loss import *


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


class Model(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, word_num, query_num, entity_num,
                 word_embedding_size, entity_embedding_size,
                 head_num,
                 convolution_num,
                 l2):
        super().__init__()
        self.graph = graph
        self.word_num = word_num
        self.query_num = query_num
        self.entity_num = entity_num

        self.word_embedding_size = word_embedding_size
        self.entity_embedding_size = entity_embedding_size
        self.convolution_num = convolution_num

        self.l2 = l2

        # self.doc_embedding = nn.Sequential(SelfAttentionLayer(word_embedding_size))
        self.doc_embedding = nn.Sequential(MultiHeadSelfAttention(input_dim=word_embedding_size,
                                                                  hidden_dim=word_embedding_size // head_num,
                                                                  # hidden_dim=word_embedding_size,
                                                                  head_num=head_num),
                                           # FeedForward(word_embedding_size, 4 * word_embedding_size),
                                           Mean(dim=1))
        # self.doc_embedding = nn.Sequential(nn.Linear(word_embedding_size, word_embedding_size),
        #                                    nn.ELU(),
        #                                    nn.Linear(word_embedding_size, word_embedding_size),
        #                                    nn.ELU(),
        #                                    Mean(dim=1))
        # self.query_translation = nn.Sequential(Mean(dim=1),
        #                                        nn.Linear(word_embedding_size, entity_embedding_size),
        #                                        nn.ELU())
        self.query_translation = nn.Sequential(MultiHeadSelfAttention(input_dim=word_embedding_size,
                                                                      hidden_dim=word_embedding_size // head_num,
                                                                      head_num=head_num),
                                               Mean(dim=1),
                                               nn.Linear(word_embedding_size, entity_embedding_size))

        self.word_embedding_layer = nn.Embedding(word_num, word_embedding_size, padding_idx=0)
        self.entity_embedding_layer = nn.Embedding(entity_num, entity_embedding_size)

        self.reset_parameters()

    def init_graph(self):
        self.graph.nodes['word'].data['w'] = self.word_embedding_layer.weight
        self.graph.nodes['entity'].data['l'] = self.entity_embedding_layer.weight
        self.graph.nodes['entity'].data['deg_p'] = self.graph.in_degrees(
            etype='profiles').float().unsqueeze(dim=-1).clamp(min=1)
        self.graph.nodes['entity'].data['deg_i'] = torch.add(
            self.graph.out_degrees(etype='purchased').float().unsqueeze(dim=-1),
            self.graph.out_degrees(etype='purchased_by').float().unsqueeze(dim=-1)).clamp(min=1)
        # with torch.no_grad():
        #     self.graph.nodes['entity'].data['e'] = torch.zeros(self.entity_num,
        #                                                        self.word_embedding_size +
        #                                                        self.entity_embedding_size).cuda()

    def apply_word2vec(self, word2vec: torch.Tensor = None):
        nn.init.normal_(self.word_embedding_layer.weight, 0, 0.1)
        if self.word_embedding_layer.padding_idx is not None:
            with torch.no_grad():
                self.word_embedding_layer.weight[self.word_embedding_layer.padding_idx].fill_(0)
        # self.word_embedding_layer.load_state_dict({'weight': word2vec})

    def reset_parameters(self):
        # nn.init.normal_(self.entity_embedding_layer.weight, 0, 1)

        for layers in [self.doc_embedding, self.query_translation]:
            for layer in layers:
                if isinstance(layer, MultiHeadSelfAttention):
                    layer.reset_parameters()
                elif isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)

    def __get_query_embedding(self):
        with self.graph.local_scope():

            def reduce_w_q(queries):
                query_translation = self.query_translation(queries.mailbox['m'])
                doc_embedding = self.doc_embedding(queries.mailbox['m'])
                return {'l': query_translation, 'h': doc_embedding}
                # return {'l': query_translation}

            self.graph.update_all(message_func=fn.copy_u('w', 'm'),
                                  reduce_func=reduce_w_q,
                                  etype=('word', 'composes', 'query'))
            # words->query Apply Function
            query_embeddings = torch.cat([self.graph.nodes['query'].data['l'],
                                          self.graph.nodes['query'].data['h']], dim=-1)
            # query_embeddings = self.graph.nodes['query'].data['l']
        return query_embeddings

    def graph_propagation(self):
        # ---------------Tier 1---------------
        with self.graph.local_scope():
            self.graph.update_all(message_func=fn.copy_u('w', 'm'),
                                  reduce_func=lambda reviews:
                                  {
                                      'h': self.doc_embedding(reviews.mailbox['m'])
                                  },
                                  etype=('word', 'composes', 'review'))

            self.graph.nodes['query'].data['e_0'] = self.__get_query_embedding()

            # ---------------Tier 2---------------
            self.graph.update_all(message_func=fn.copy_u('h', 'm'),
                                  reduce_func=fn.sum('m', 'h'),
                                  etype='profiles')
            entity_embeddings = self.graph.nodes['entity'].data['h'] / self.graph.nodes['entity'].data['deg_p']
            # shape: (batch, entity_size + word_size)
            self.graph.nodes['entity'].data['e_0'] = torch.cat([self.graph.nodes['entity'].data['l'],
                                                                entity_embeddings], dim=-1)

            # # shape: (batch, entity_size)
            # self.graph.nodes['entity'].data['e_0'] = self.graph.nodes['entity'].data['l']

            def msg_u_i(edges: udf.EdgeBatch):
                q = self.graph.nodes['query'].data['e_0'][edges.data['q_id']]
                return {'m': edges.src['e_%d' % k] + (q / (edges.src['deg_i'] ** 0.5))}

            # ---------------Tier 3---------------
            entities = self.graph.nodes['entity']
            for k in range(0, self.convolution_num):
                entities.data['e_%d_msg' % k] = entities.data['e_%d' % k] / (entities.data['deg_i'] ** 0.5)
                # user -> item
                self.graph.update_all(message_func=msg_u_i,
                                      reduce_func=fn.sum('m', 'e_%d' % (k + 1)),
                                      etype='purchased')
                # item -> user
                self.graph.update_all(message_func=fn.copy_u('e_%d_msg' % k, 'm'),
                                      reduce_func=fn.sum('m', 'e_%d' % (k + 1)),
                                      etype='purchased_by')
                entities.data['e_%d' % (k + 1)] /= (entities.data['deg_i'] ** 0.5)

            # ---------------Layer Combination---------------
            entity_embeddings = torch.stack(list(map(lambda i: self.graph.nodes['entity'].data['e_%d' % i],
                                                     range(0, self.convolution_num+1))))
            # shape: (k, batch, entity_size + word_size)
            entity_embeddings = torch.sum(entity_embeddings, dim=0) / (self.convolution_num + 1)
            # shape: (batch, entity_size + word_size)

        self.graph.nodes['entity'].data['e'] = entity_embeddings

    def regularization_loss(self):
        l2_norm = 0
        for weight in self.doc_embedding.parameters():
            l2_norm += weight.norm()
        for weight in self.query_translation.parameters():
            l2_norm += weight.norm()
        l2_norm += self.word_embedding_layer.weight.norm() + self.entity_embedding_layer.weight.norm()
        return self.l2 * l2_norm

    def forward(self, users, items, query_words, mode='train', neg_items=None):
        """
        Parameters
        -----
        users: torch.LongTensor
            shape: (batch, )
        items: torch.LongTensor
            shape: (batch, )
        query_words: torch.LongTensor
            shape: (batch, seq)
        mode: bool
            one of ['train', 'test', 'output_embedding']
        neg_items: torch.LongTensor
            shape: (batch, k)
        """
        if mode == 'output_embedding':
            # item_embeddings = self.graph.nodes['entity'].data['e'][items]
            item_embeddings = self.graph.nodes['entity'].data['e'][items][:, :self.entity_embedding_size]
            return item_embeddings

        if mode == 'train':
            self.graph_propagation()

        user_embeddings = self.graph.nodes['entity'].data['e'][users]
        query_embeddings = self.word_embedding_layer(query_words)  # shape: (batch, seq, word_embedding_size)
        query_translation = self.query_translation(query_embeddings)

        personalized_model = user_embeddings[:, :self.entity_embedding_size] + query_translation

        # query_embeddings = self.doc_embedding(query_embeddings)  # shape: (batch, embedding_size)
        # query_embeddings = torch.cat([query_translation,
        #                               query_embeddings], dim=1)
        # personalized_model = user_embeddings + query_embeddings

        if mode == 'train':
            item_embeddings = self.graph.nodes['entity'].data['e'][items]
            # neg_user_embeddings = self.graph.nodes['entity'].data['e'][neg_users]
            neg_item_embeddings = self.graph.nodes['entity'].data['e'][neg_items]
            # return hem_loss(personalized_model, item_embeddings, neg_item_embeddings) + self.regularization_loss()

            search_loss = hem_loss(pred=personalized_model,
                                   pos=item_embeddings[:, :self.entity_embedding_size],
                                   negs=neg_item_embeddings[:, :, :self.entity_embedding_size])
            user_loss = log_loss(pred=user_embeddings[:, :self.entity_embedding_size],
                                 pos=user_embeddings[:, self.entity_embedding_size:])
            item_loss = log_loss(pred=item_embeddings[:, :self.entity_embedding_size],
                                 pos=item_embeddings[:, self.entity_embedding_size:])

            return search_loss + user_loss + item_loss + self.regularization_loss()
            # return search_loss + self.regularization_loss()
        elif mode == 'test':
            return personalized_model
            # return personalized_model[:, :self.entity_embedding_size]
        else:
            raise NotImplementedError

