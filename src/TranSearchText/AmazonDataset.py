from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from torch.utils.data import Dataset


class PreTrainData(Dataset):
    def __init__(self, asin_dict, doc2vec_model: Doc2Vec, neg_num):
        """ For pretraining the image and review features. """
        self.neg_num = neg_num
        self.asin_dict = asin_dict
        self.all_items = set(self.asin_dict.keys())

        # textual feture data
        self.doc2vec_model = doc2vec_model
        self.text_vec = {asin: self.doc2vec_model.docvecs[asin] for asin in self.asin_dict}
        self.features = []

    def sample_neg(self):
        """ Sample the anchor, positive, negative tuples. """
        self.features = []
        for asin in self.asin_dict:
            pos_items = self.asin_dict[asin]['positive']
            if not len(pos_items) == 0:
                for pos in pos_items:
                    neg = np.random.choice(list(
                        self.all_items - set(pos_items)),
                        self.neg_num, replace=False)
                    for n in neg:
                        self.features.append((asin, pos, n))

    def __len__(self):
        """ For each anchor item, sample neg_number items."""
        return len(self.features)

    def test(self):
        for asin in self.asin_dict:
            anchor_text = self.text_vec[asin]
            yield anchor_text, asin

    def __getitem__(self, idx):
        feature_idx = self.features[idx]
        anchor_item = feature_idx[0]
        pos_item = feature_idx[1]
        neg_item = feature_idx[2]

        anchor_text = self.text_vec[anchor_item]
        pos_text = self.text_vec[pos_item]
        neg_text = self.text_vec[neg_item]

        sample = {'anchor_text': anchor_text,
                  'pos_text': pos_text,
                  'neg_text': neg_text}
        return sample


class AmazonDataset(PreTrainData):
    def __init__(self, df, query_dict: dict, user_bought: dict, asin_dict: dict, doc2vec_model: Doc2Vec, neg_num, is_training: bool):
        """ Without pre-training, input the raw data. """
        super().__init__(asin_dict, doc2vec_model, neg_num)
        self.is_training = is_training
        self.data = df

        self.query_dict = query_dict
        self.user_bought = user_bought
        self.items = list(self.asin_dict.keys())
        self.features = []

    def sample_neg(self):
        """ Take the also_view or buy_after_viewing as negative samples. """
        for i in range(len(self.data)):
            query_vec = self.doc2vec_model.docvecs[
                self.query_dict[self.data['query_'][i]]]
            if self.is_training:
                # We tend to sample negative ones from the also_view and
                # buy_after_viewing items, if don't have enough, we then
                # randomly sample negative ones.
                asin = self.data['asin'][i]
                sample = self.asin_dict[asin]
                all_sample = sample['positive'] + sample['negative']
                negs = np.random.choice(
                    all_sample, self.neg_num, replace=False, p=sample['prob'])

                self.features.append(((self.data['userID'][i], query_vec), (asin, negs)))

            else:
                self.features.append(((self.data['userID'][i], query_vec),
                                      (self.data['reviewerID'][i], self.data['asin'][i]), self.data['query_'][i]))

    def neg_candidates(self, asin):
        a = list(set(self.items) - {asin})
        candidates = np.random.choice(a, 99, replace=False)
        candidates_text = []
        for candidate in candidates:
            candidates_text.append(self.text_vec[candidate])
        candidates_text = torch.tensor(candidates_text).cuda()
        return candidates_text

    def get_all_test(self):
        for asin in self.asin_dict:
            sample_text = self.text_vec[asin]
            yield sample_text, asin

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature_idx = self.features[idx]
        user_id = feature_idx[0][0]
        query = feature_idx[0][1]

        if self.is_training:
            pos_item = feature_idx[1][0]
            neg_items = feature_idx[1][1]

            pos_text = self.text_vec[pos_item]

            neg_text = [self.text_vec[i] for i in neg_items]
            neg_text = np.array(neg_text)

            sample = {'userID': user_id, 'query': query, 'pos_text': pos_text, 'neg_text': neg_text}

        else:
            reviewer_id = feature_idx[1][0]
            item = feature_idx[1][1]
            query_text = feature_idx[2]

            sample = {'userID': user_id, 'query': query,
                      'reviewerID': reviewer_id, 'item': item,
                      'query_text': query_text}
        return sample
