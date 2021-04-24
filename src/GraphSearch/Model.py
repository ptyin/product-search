import torch
from torch import nn
import dgl
from dgl import udf
import dgl.function as fn

from GraphSearch.utils.self_attention.MultiHeadSelfAttention import MultiHeadSelfAttention
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
                 convolution_num):
        super().__init__()
        self.graph = graph
        self.word_num = word_num
        self.query_num = query_num
        self.entity_num = entity_num

        self.word_embedding_size = word_embedding_size
        self.entity_embedding_size = entity_embedding_size
        self.convolution_num = convolution_num

        # self.doc_embedding = nn.Sequential(SelfAttentionLayer(word_embedding_size))
        self.doc_embedding = nn.Sequential(MultiHeadSelfAttention(input_dim=word_embedding_size,
                                                                  hidden_dim=word_embedding_size // head_num,
                                                                  head_num=head_num),
                                           # FeedForward(word_embedding_size, 4 * word_embedding_size),
                                           Mean(dim=1))
        # self.self_attention = MultiHeadSelfAttention(word_embedding_size, head_num)
        # self.feed_forward = FeedForward(word_embedding_size, 4 * word_embedding_size)

        self.word_embedding_layer = nn.Embedding(word_num, word_embedding_size, padding_idx=0)
        self.query_embedding_layer = nn.Embedding(query_num, entity_embedding_size)
        self.entity_embedding_layer = nn.Embedding(entity_num, entity_embedding_size)
        # self.personalized_factor = nn.Parameter(torch.tensor(0.0))

        self.reset_parameters()

    def init_graph(self):
        self.graph.nodes['word'].data['w'] = self.word_embedding_layer.weight
        # self.graph.nodes['query'].data['l'] = self.query_embedding_layer.weight
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

    def reset_parameters(self):
        self.word_embedding_layer.reset_parameters()
        self.entity_embedding_layer.reset_parameters()
        for layer in self.doc_embedding:
            layer.reset_parameters()
        # nn.init.uniform_(self.personalized_factor)

    def __get_query_embedding(self):
        with self.graph.local_scope():
            self.graph.update_all(message_func=fn.copy_u('w', 'm'),
                                  reduce_func=lambda reviews:
                                  {
                                      'h': self.doc_embedding(reviews.mailbox['m'])
                                  },
                                  etype=('word', 'composes', 'query'))
            # words->query Apply Function
            # query_embeddings = torch.cat([self.graph.nodes['query'].data['l'],
            #                               self.graph.nodes['query'].data['h']], dim=-1)
            query_embeddings = torch.cat([torch.zeros(self.query_num, self.entity_embedding_size).cuda(),
                                          self.graph.nodes['query'].data['h']], dim=-1)
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

            def msg_u_i(edges: udf.EdgeBatch):
                # print(edges.src['e_%d' % k].shape,
                #       edges.data['q_id'].shape,
                #       self.graph.nodes['query'].data['e_0'][edges.data['q_id']].shape)
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

    def forward(self, users, items, query_words, mode='train', negs=None):
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
        negs: torch.LongTensor
            shape: (batch, )
        """
        if mode == 'output_embedding':
            item_embeddings = self.graph.nodes['entity'].data['e'][items]
            return item_embeddings

        if mode == 'train':
            self.graph_propagation()

        batch_size = len(users)
        user_embeddings = self.graph.nodes['entity'].data['e'][users]
        query_embeddings = self.word_embedding_layer(query_words)  # shape: (batch, seq, word_embedding_size)
        query_embeddings = self.doc_embedding(query_embeddings)  # shape: (batch, embedding_size)
        query_embeddings = torch.cat([torch.zeros(batch_size, self.entity_embedding_size).cuda(),
                                      query_embeddings], dim=1)
        personalized_model = user_embeddings + query_embeddings

        if mode == 'train':
            item_embeddings = self.graph.nodes['entity'].data['e'][items]
            neg_embeddings = self.graph.nodes['entity'].data['e'][negs]
            # return personalized_model, item_embeddings, neg_embeddings
            return bpr_loss(personalized_model, item_embeddings, neg_embeddings)
        elif mode == 'test':
            # return personalized_model, item_embeddings
            return personalized_model
        else:
            raise NotImplementedError

