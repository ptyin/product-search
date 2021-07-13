import os
import sys
import random
import json
import argparse
import collections
import itertools

import numpy as np
import pandas as pd
from ast import literal_eval
from gensim.models import doc2vec

parser = argparse.ArgumentParser()

parser.add_argument("--embedding_size",
                    type=int,
                    default=512,
                    help="doc dimension.")
parser.add_argument("--window_size",
                    type=int,
                    default=3,
                    help="sentence window size.")
# parser.add_argument("--img_feature_file",
#     type=str,
#     default="data/image_features_Musical_Instruments.b",
#     help="the raw image feature file")

parser.add_argument('--dataset', type=str, default='Musical_Instruments')
parser.add_argument('--main_path', type=str, default='/disk/yxk/data/cold_start/')
parser.add_argument('--stop_file', type=str, default='../../seq_utils/TranSearch/stopwords.txt')
parser.add_argument('--processed_path', type=str,
                    default='/disk/yxk/processed/cold_start/ordinary/Musical_Instruments/')

FLAGS = parser.parse_args()

# --------------------------- PREPARE DATA ---------------------------
full_path = os.path.join(FLAGS.processed_path, '{}_full.csv'.format(FLAGS.dataset))
full_data = pd.read_csv(full_path)
full_data.query_ = full_data.query_.apply(literal_eval)
full_data.reviewText = full_data.reviewText.apply(literal_eval)
asin_set = set(full_data.asin.unique())
# img_path = os.path.join(FLAGS.main_path, FLAGS.img_feature_file)

# gather reviews to same asins
raw_doc = collections.defaultdict(list)
for k, v in zip(full_data.asin, full_data.reviewText):
    raw_doc[k].append(v)

# concatenate the reviews together
for k in raw_doc.keys():
    m = [i for i in raw_doc[k]]
    m = list(itertools.chain.from_iterable(m))
    raw_doc[k] = m

# for query, it's hard to tag, so we just random tag them
query_idx, query_dict = 0, {}
for q in full_data['query_']:
    if repr(q) not in query_dict:
        query_dict[repr(q)] = query_idx
        raw_doc[query_idx] = q
        query_idx += 1

# --------------------------- MODEL TRAINING ---------------------------
analyzed_doc = collections.namedtuple(
    'AnalyzedDocument', 'words tags')
docs = [analyzed_doc(raw_doc[d], [d]) for d in raw_doc.keys()]

alpha_val = 0.025
min_alpha_val = 1e-4
passes = 40

alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)
model = doc2vec.Doc2Vec(
    min_count=2,
    workers=4,
    epochs=40,
    vector_size=FLAGS.embedding_size,
    window=FLAGS.window_size)

model.build_vocab(docs)  # Building vocabulary
for epoch in range(passes):
    random.shuffle(docs)

    model.alpha, model.min_alpha = alpha_val, alpha_val
    model.train(docs, total_examples=len(docs), epochs=model.epochs)
    alpha_val -= alpha_delta
    print('epochs:', epoch)

# --------------------------- SAVE TO DISK ---------------------------

df = full_data
users = df['userID'].tolist()
json.dump(users, open(os.path.join(FLAGS.processed_path, '{}_users.json'.format(FLAGS.dataset)), 'w'))
products = df['asin'].tolist()
item_map = dict(zip(products, range(len(products))))
json.dump(item_map, open(os.path.join(FLAGS.processed_path, '{}_item_map.json'.format(FLAGS.dataset)), 'w'))

# for metaFilter in ['TrainSupport', 'TrainQuery', 'TestSupport', 'TestQuery']:
#     target_df = df[df["metaFilter"] == metaFilter]
#     target_df = target_df.set_index(['userID'])
#     users = target_df.index.unique()
#     target_data = {}
#
#     for user in users:
#         # ---------------------------------meta---------------------------------
#         target_items = pd.Series(target_df.loc[user, 'asin'], dtype=str).map(item_map).tolist()
#         target_queries = pd.Series(target_df.loc[user, 'query_'], dtype=str).map(lambda query: model.docvecs[query_dict[query]].tolist()).tolist()
#         target_data[user] = {'items': target_items, 'queries': target_queries}
#
#     json.dump(target_data, open(os.path.join(FLAGS.processed_path, '{}_{}_data.json'.format(FLAGS.dataset, metaFilter)),
#                                 'w'))

doc2model_path = FLAGS.processed_path + '{}_doc2model'.format(FLAGS.dataset)
query_path = FLAGS.processed_path + '{}_query.json'.format(FLAGS.dataset)
# img_feature_path = FLAGS.processed_path + '{}_img_feature.npy'.format(FLAGS.dataset)


model.save(doc2model_path)

json.dump(query_dict, open(query_path, 'w'))
print("The query number is %d." % len(query_dict))
