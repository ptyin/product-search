import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np


import dgl


class AmazonDataset(Dataset):
    def __init__(self, df, users, item_map: dict, asin_dict):
        self.df = df
        self.users = users
        self.item_map = item_map
        self.asin_dict = asin_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        series = self.df.loc[index]
        user = torch.tensor(series['userID'], dtype=torch.long).cuda()
        asin = series['asin']
        item = torch.tensor(self.item_map[asin], dtype=torch.long).cuda()
        neg = torch.tensor(self.sample_neg(asin), dtype=torch.long).cuda()
        query_words = torch.tensor(eval(series['queryWords']), dtype=torch.long).cuda()

        return user, item, neg, query_words

    def sample_neg(self, asin):
        """ Take the also_view or buy_after_viewing as negative samples. """
        # We tend to sample negative ones from the also_view and
        # buy_after_viewing items, if don't have enough, we then
        # randomly sample negative ones.

        sample = self.asin_dict[asin]
        all_sample = sample['positive'] + sample['negative']
        neg = np.random.choice(all_sample, 1, replace=False, p=sample['prob'])
        if neg[0] not in self.item_map:
            neg = np.random.choice(list(set(self.item_map.keys()) - {asin}), 1, replace=False)
        return self.item_map[neg[0]]

    def neg_candidates(self, item: torch.LongTensor):
        a = list(range(len(self.users), item.item())) + list(range(item.item() + 1, len(self.users) + len(self.item_map)))
        candidates = torch.tensor(np.random.choice(a, 99, replace=False), dtype=torch.long).cuda()
        return candidates

    @staticmethod
    def clip_words(full_df: pd.DataFrame):
        def clip(review):
            review = eval(review)
            max_word_num = 15
            if len(review) > max_word_num:
                return np.random.choice(review, max_word_num, replace=False).tolist()
            else:
                return review
        full_df['reviewWords'] = full_df['reviewWords'].map(clip)

    @staticmethod
    def construct_graph(df: pd.DataFrame, word_num: int):
        # TODO, clip word num and review num
        users = df['userID'].unique()
        items = df['asin'].unique()
        item_map = dict(zip(items, range(len(users), len(users) + len(items))))
        query_map = {}
        current_review_id = 0

        tier1_src_r, tier1_des_r,  = [], []
        tier1_src_q, tier1_des_q = [], []
        tier2_src, tier2_des = [], []
        tier3_u, tier3_i = [], []

        e_data = []

        # words, reviews, users, items = [], [], [], []

        for index, series in df.iterrows():
            if series['filter'] == 'Train':
                # ------------------------Tier 1------------------------
                # ********word->query********
                current_words = tuple(eval(series['queryWords']))
                if current_words not in query_map:
                    query_map[current_words] = len(query_map)
                current_query_id = query_map[current_words]
                tier1_src_q += current_words
                tier1_des_q += [current_query_id] * len(current_words)

                if len(eval(series['reviewText'])) != 0:
                    # ********word->review********
                    current_words = series['reviewWords']
                    tier1_src_r += current_words
                    tier1_des_r += [current_review_id] * len(current_words)

                    # ------------------------Tier 2------------------------
                    # ********review->entity********
                    tier2_src += [current_review_id, current_review_id]
                    tier2_des += [series['userID'], item_map[series['asin']]]

                    current_review_id += 1

                # ------------------------Tier 3------------------------
                # ********user<->item********
                tier3_u.append(series['userID'])
                tier3_i.append(item_map[series['asin']])
                e_data.append(current_query_id)

                # current_query_id += 1

        graph_data = {('word', 'composes', 'review'): (tier1_src_r, tier1_des_r),
                      ('word', 'composes', 'query'): (tier1_src_q, tier1_des_q),
                      ('review', 'profiles', 'entity'): (tier2_src, tier2_des),
                      ('entity', 'purchased', 'entity'): (tier3_u, tier3_i),
                      ('entity', 'purchased_by', 'entity'): (tier3_i, tier3_u)}
        num_nodes_dict = {'word': word_num,
                          'query': len(query_map), 'review': current_review_id,
                          'entity': len(users)+len(items)}
        graph: dgl.DGLHeteroGraph = dgl.heterograph(graph_data, num_nodes_dict)
        graph.edges['purchased'].data['q_id'] = torch.tensor(e_data, dtype=torch.long)

        # plot_meta(graph)
        # plot(graph)
        return users, item_map, query_map, graph

    @staticmethod
    def collate_fn(batch):
        entities = []
        query_words = []
        for sample in batch:
            entities.append(sample[:-1])
            query_words.append(sample[-1])
        entity_result = default_collate(entities)  # shape:(3, batch)
        query_result = pad_sequence(query_words, batch_first=True, padding_value=0)  # shape: (batch, seq)
        return (*entity_result, query_result)
