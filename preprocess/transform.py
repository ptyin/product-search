import argparse
import gzip
import operator
import json
import os
import ast
import pandas as pd


class Transform:
    def __init__(self, review_file, output_path, meta_file):
        self.word_map = {'': 0}
        self.user_map = {}
        self.product_map = {}
        self.query_map = {'': 0, }

        self.word_list = ['']
        self.user_list = []
        self.product_list = []
        self.query_list = ['']

        self.train_queries = []
        self.test_queries = []

        self.review_file = review_file
        self.min_count = 5
        self.output_path = os.path.join(output_path, 'seq_min_count' + str(self.min_count) + '/')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.meta_file = meta_file
        self.split_output_path = os.path.join(self.output_path, 'seq_query_split/')
        if not os.path.exists(self.split_output_path):
            os.makedirs(self.split_output_path)

    def index_and_filter_review_file(self):
        """
            Index and filter review, from index_and_filter_review_file.py
        """
        min_count = self.min_count
        output_path = self.output_path

        # read all words, users, products
        word_count_map = {}
        user_set = set()
        product_set = set()
        with gzip.open(self.review_file, 'r') as g:
            for l in g:
                l = eval(l)
                user = l['reviewerID']
                product = l['asin']
                review_text = l['reviewText']
                summary = l['summary']
                user_set.add(user)
                product_set.add(product)
                for term in review_text.strip().split(' '):
                    if term not in word_count_map:
                        word_count_map[term] = 0
                    word_count_map[term] += 1
                for term in summary.strip().split(' '):
                    if term not in word_count_map:
                        word_count_map[term] = 0
                    word_count_map[term] += 1

        # filter vocabulary by min_count
        delete_key = set()
        for key in word_count_map:
            if word_count_map[key] < min_count:
                delete_key.add(key)
        # output word, user, product indexes
        word_list = list(set(word_count_map.keys()) - delete_key)
        # with gzip.open(output_path + 'vocab.txt.gz', 'wt') as fout:
        #     for word in word_list:
        #         fout.write(word + '\n')
        self.user_list = list(user_set)
        with gzip.open(output_path + 'users.txt.gz', 'wt') as fout:
            for user in self.user_list:
                fout.write(user + '\n')
        self.product_list = list(product_set)
        with gzip.open(output_path + 'product.txt.gz', 'wt') as fout:
            for product in self.product_list:
                fout.write(product + '\n')

        # read and output indexed reviews 将一个列表转换成一个字典，key是集合中的元素，value是序号
        def index_set(s):
            i = 0
            s_map = {}
            for key in s:
                s_map[key] = i
                i += 1
            return s_map

        word_map = index_set(word_list)
        self.user_map = index_set(self.user_list)
        self.product_map = index_set(self.product_list)
        user_review_seq = {}  # recording the sequence of user reviews in time
        count_valid_review = 0
        with gzip.open(output_path + 'review_u_p.txt.gz', 'wt') as fout_u_p:
            with gzip.open(output_path + 'review_id.txt.gz', 'wt') as fout_id, \
                    gzip.open(output_path + 'review_rating.txt.gz', 'wt') as fout_rating:
                with gzip.open(self.review_file, 'r') as g:
                    index = 0
                    for l in g:
                        l = eval(l)
                        user = l['reviewerID']
                        product = l['asin']
                        review_text = l['reviewText']
                        summary = l['summary']
                        rating = l['overall']
                        time = l['unixReviewTime']
                        count_words = 0
                        for term in summary.strip().split(' '):
                            if term in word_map:
                                # fout_text.write(word_map[term] + ' ')
                                count_words += 1
                        for term in review_text.strip().split(' '):
                            if term in word_map:
                                # fout_text.write(word_map[term] + ' ')
                                count_words += 1

                        if count_words > 0:
                            if user not in user_review_seq:
                                user_review_seq[user] = []
                            user_review_seq[user].append((count_valid_review, time))
                            # fout_text.write('\n')
                            fout_u_p.write(str(self.user_map[user]) + ' ' + str(self.product_map[product]) + '\n')
                            fout_id.write('line_' + str(index) + '\n')
                            fout_rating.write(str(rating))
                            count_valid_review += 1
                        index += 1

        # Sort each user's reviews according to time and output to files
        review_loc_time_list = [[] for _ in range(count_valid_review)]
        with gzip.open(output_path + 'u_r_seq.txt.gz', 'wt') as fout:
            for user in self.user_list:
                review_time_list = user_review_seq[user]
                user_review_seq[user] = sorted(review_time_list, key=operator.itemgetter(1))
                fout.write(' '.join([str(x[0]) for x in user_review_seq[user]]) + '\n')
                for i in range(len(user_review_seq[user])):
                    review_id = user_review_seq[user][i][0]
                    time = user_review_seq[user][i][1]
                    review_loc_time_list[review_id] = [i, time]

    def match_and_create_knowledge(self):
        """
            Gather knowledge from meta data, from match_with_meta_knowledge.py
        """
        data_path = self.output_path
        meta_path = self.meta_file

        # read needed product ids
        product_ids = []
        with gzip.open(data_path + 'product.txt.gz', 'rt') as fin:
            for line in fin:
                product_ids.append(line.strip())
        product_indexes = dict([(product_ids[i], i) for i in range(len(product_ids))])

        # match with meta data
        brand_vocab, brand_list = {}, []
        category_vocab, category_list = {}, []
        related_product_vocab, related_product_list = {}, []
        know_dict = {
            'also_bought': [[] for _ in range(len(product_ids))],
            'also_viewed': [[] for _ in range(len(product_ids))],
            'bought_together': [[] for _ in range(len(product_ids))],
            'brand': [[] for _ in range(len(product_ids))],
            'categories': [[] for _ in range(len(product_ids))]
        }
        count_dict = {
            'also_bought': 0,
            'also_viewed': 0,
            'bought_together': 0,
            'brand': 0,
            'categories': 0
        }
        # read meta_data
        with gzip.open(meta_path, 'rt') as fin:
            for line in fin:
                meta = ast.literal_eval(line)
                if meta['asin'] in product_indexes:
                    pidx = product_indexes[meta['asin']]
                    if 'related' in meta:
                        related = meta['related']
                        if 'also_bought' in related:
                            for asin in related['also_bought']:
                                if asin not in related_product_vocab:
                                    related_product_vocab[asin] = len(related_product_list)
                                    related_product_list.append(asin)
                            know_dict['also_bought'][pidx] = [related_product_vocab[asin] for asin in
                                                              related['also_bought']]
                            count_dict['also_bought'] += len(know_dict['also_bought'][pidx])
                        # also view
                        if 'also_viewed' in related:
                            for asin in related['also_viewed']:
                                if asin not in related_product_vocab:
                                    related_product_vocab[asin] = len(related_product_list)
                                    related_product_list.append(asin)
                            know_dict['also_viewed'][pidx] = [related_product_vocab[asin] for asin in
                                                              related['also_viewed'] if asin in product_indexes]
                            count_dict['also_viewed'] += len(know_dict['also_viewed'][pidx])
                        # bought together
                        if 'bought_together' in related:
                            for asin in related['bought_together']:
                                if asin not in related_product_vocab:
                                    related_product_vocab[asin] = len(related_product_list)
                                    related_product_list.append(asin)
                            know_dict['bought_together'][pidx] = [related_product_vocab[asin] for asin in
                                                                  related['bought_together'] if asin in product_indexes]
                            count_dict['bought_together'] += len(know_dict['bought_together'][pidx])
                    # brand
                    if 'brand' in meta:
                        if meta['brand'] not in brand_vocab:
                            brand_vocab[meta['brand']] = len(brand_list)
                            brand_list.append(meta['brand'])
                        know_dict['brand'][pidx] = [brand_vocab[meta['brand']]]
                        count_dict['brand'] += 1
                    # categories
                    if 'categories' in meta:
                        categories_set = set()
                        for category_tree in meta['categories']:
                            for category in category_tree:
                                if category not in category_vocab:
                                    category_vocab[category] = len(category_list)
                                    category_list.append(category)
                                categories_set.add(category_vocab[category])
                        know_dict['categories'][pidx] = list(categories_set)
                        count_dict['categories'] += len(know_dict['categories'][pidx])

        fout_dict = {
            'also_bought': gzip.open(data_path + 'also_bought_p_p.txt.gz', 'wt'),
            'also_viewed': gzip.open(data_path + 'also_viewed_p_p.txt.gz', 'wt'),
            'bought_together': gzip.open(data_path + 'bought_together_p_p.txt.gz', 'wt'),
            'brand': gzip.open(data_path + 'brand_p_b.txt.gz', 'wt'),
            'categories': gzip.open(data_path + 'category_p_c.txt.gz', 'wt')
        }
        for key in fout_dict:
            for i in range(len(product_ids)):
                # Write to files
                str_list = [str(x) for x in know_dict[key][i]]
                fout_dict[key].write(' '.join(str_list) + '\n')
            fout_dict[key].close()
        with gzip.open(data_path + 'related_product.txt.gz', 'wt') as fout:
            for asin in related_product_list:
                fout.write(asin + '\n')
        with gzip.open(data_path + 'brand.txt.gz', 'wt') as fout:
            for brand in brand_list:
                fout.write(brand + '\n')
        with gzip.open(data_path + 'category.txt.gz', 'wt') as fout:
            for category in category_list:
                fout.write(category + '\n')
        with open(data_path + 'knowledge_statistics.txt', 'wt') as fout:
            fout.write('Total Brand num %d\n' % len(brand_list))
            fout.write('Total Category num %d\n' % len(category_list))
            fout.write('Avg also_bought per product %.3f\n' % (float(count_dict['also_bought']) / len(product_ids)))
            fout.write('Avg also_view per product %.3f\n' % (float(count_dict['also_viewed']) / len(product_ids)))
            fout.write(
                'Avg bought_together per product %.3f\n' % (float(count_dict['bought_together']) / len(product_ids)))
            fout.write('Avg brand per product %.3f\n' % (float(count_dict['brand']) / len(product_ids)))
            fout.write('Avg category per product %.3f\n' % (float(count_dict['categories']) / len(product_ids)))

    def read_from_csv(self, path):
        df = pd.read_csv(path)
        # users = list(map(lambda user: self.user_map[user], df['reviewerID'].tolist()))
        # products = list(map(lambda product: self.product_map[product], df['asin'].tolist()))
        # reviews = list(map(lambda review: eval(review), df['reviewText'].tolist()))
        # queries = list(map(lambda query: eval(query), df['query_'].tolist()))
        df_users = df['reviewerID'].tolist()
        df_products = df['asin'].tolist()
        df_reviews = df['reviewText'].tolist()
        df_queries = df['query_'].tolist()
        users = []
        products = []
        reviews = []
        queries = []

        for i in range(len(df_users)):
            if len(eval(df_reviews[i])) != 0:
                users.append(self.user_map[df_users[i]])
                products.append(self.product_map[df_products[i]])
                reviews.append(eval(df_reviews[i]))
                queries.append(eval(df_queries[i]))
        return users, products, reviews, queries

    def build_vocab_map(self, reviews, queries):
        for review in reviews:
            for word in review:
                if word not in self.word_map:
                    self.word_map[word] = len(self.word_map)  # build word->index map
                    self.word_list.append(word)  # build index->word map
        for query in queries:
            for word in query:
                if word not in self.word_map:
                    self.word_map[word] = len(self.word_map)  # build word->index map
                    self.word_list.append(word)  # build index->word map

    def generate_vocab_file(self):
        """
            path: *_full.csv
            generate indexed words file -- vocab.txt.gz
        """
        with gzip.open(os.path.join(self.output_path, 'vocab.txt.gz'), 'wt') as fout:
            for word in self.word_list:
                fout.write(word + '\n')

    def generate_queries_file(self, queries):
        with gzip.open(os.path.join(self.split_output_path, 'query.txt.gz'), 'wt') as query_output:
            query_output.write('0\n')
            for query in queries:
                query = ' '.join(list(map(lambda word: str(self.word_map[word]), query)))
                if query not in self.query_map:
                    self.query_map[query] = len(self.query_map)  # build query->index map
                    query_output.write(query+' ')
                    query_output.write('\n')
                    self.query_list.append(query)  # build index->query map

    def generate_review_file(self, flag, users, products, reviews):
        """
            flag = train or test
        """
        with gzip.open(os.path.join(self.split_output_path, flag + '.txt.gz'), 'wt') as output:
            for i in range(len(users)):
                output.write(str(users[i]) + '\t' + str(products[i]) + '\t')
                output.write(' '.join(list(map(lambda word: str(self.word_map[word]), reviews[i]))))
                output.write('\n')

    def generate_query_idx_file(self, flag, products, queries):
        with gzip.open(os.path.join(self.split_output_path, flag + '_query_idx.txt.gz'), 'wt') as output:
            for product in self.product_list:
                if self.product_map[product] in products:
                    category = map(lambda word: str(self.word_map[word]), queries[products.index(self.product_map[product])])
                else:
                    category = []
                category = ' '.join(list(category))
                output.write(str(self.query_map[category]) + '\n')
                if flag == "train":
                    # build train set product index -> queries map
                    self.train_queries.append([self.query_map[category]])
                elif flag == "test":
                    # build test set product index -> queries map
                    self.test_queries.append([self.query_map[category]])

    def output_qrels_json_query(self, user_product_map, flag):
        if flag == "train":
            product_query = self.train_queries
        else:
            product_query = self.test_queries
        qrel_file = os.path.join(self.split_output_path, flag+'.qrels')
        json_query_file = os.path.join(self.split_output_path, flag+'_query.json')
        json_queries = []
        appeared_qrels = {}
        with open(qrel_file, 'wt') as fout:
            for u_idx in user_product_map:
                user_id = self.user_list[u_idx]
                if user_id not in appeared_qrels:
                    appeared_qrels[user_id] = {}
                for product_idx in user_product_map[u_idx]:
                    product_id = self.product_list[product_idx]
                    if product_id not in appeared_qrels[user_id]:
                        appeared_qrels[user_id][product_id] = set()
                    # check if has query
                    for q_idx in product_query[product_idx]:
                        if q_idx in appeared_qrels[user_id][product_id]:
                            continue
                        appeared_qrels[user_id][product_id].add(q_idx)
                        fout.write(user_id + '_' + str(q_idx) + ' 0 ' + product_id + ' 1 ' + '\n')
                        json_q = {'number': user_id + '_' + str(q_idx), 'text': []}
                        json_q['text'].append('#combine(')
                        for v_i in self.query_list[q_idx].strip().split(' '):
                            if len(v_i) > 0:
                                json_q['text'].append(self.word_list[int(v_i)])
                        json_q['text'].append(')')
                        json_q['text'] = ' '.join(json_q['text'])
                        json_queries.append(json_q)
        with open(json_query_file, 'wt') as fout:
            output_json = {'mu': 1000, 'queries': json_queries}
            json.dump(output_json, fout, sort_keys=True, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir',
                        type=str,
                        default='/disk/yxk/processed/cold_start/ordinary/Musical_Instruments/',
                        help="csv file generated by process.py")
    parser.add_argument('--review_file',
                        type=str,
                        default='/disk/yxk/data/cold_start/reviews_Musical_Instruments_5.json.gz',
                        help="5 core review file")
    parser.add_argument('--output_dir',
                        type=str,
                        default='/disk/yxk/transformed/cold_start/ordinary/indexed_data/',
                        help="output directory")
    parser.add_argument('--meta_file',
                        type=str,
                        default='/disk/yxk/data/cold_start/meta_Musical_Instruments.json.gz',
                        help="meta data file for the corresponding review file")
    parser.add_argument('--dataset',
                        type=str,
                        default='Musical_Instruments',
                        help="Amazon Dataset(Automotive/Cell_Phones_and_Accessories/Clothing_Shoes_and_Jewelry"
                             "/Digital_Music/Electronics/Toys_and_Games)")
    FLAGS = parser.parse_args()
    full_csv = os.path.join(FLAGS.csv_dir, '{}_full.csv'.format(FLAGS.dataset))
    train_csv = os.path.join(FLAGS.csv_dir, '{}_train.csv'.format(FLAGS.dataset))
    test_csv = os.path.join(FLAGS.csv_dir, '{}_test.csv'.format(FLAGS.dataset))

    transform = Transform(FLAGS.review_file, FLAGS.output_dir, FLAGS.meta_file)
    # index_and_filter_review_file
    print('Index and filter review')
    transform.index_and_filter_review_file()
    # match_and_create_knowledge
    # print('Gather knowledge from meta data')
    # transform.match_and_create_knowledge()

    users, products, reviews, queries = transform.read_from_csv(full_csv)
    train_users, train_products, train_reviews, train_queries = transform.read_from_csv(train_csv)
    test_users, test_products, test_reviews, test_queries = transform.read_from_csv(test_csv)

    print('generate vocab.txt.gz')
    transform.build_vocab_map(train_reviews, train_queries)
    transform.build_vocab_map(test_reviews, test_queries)
    transform.generate_vocab_file()  # vocab.txt.gz
    print('generate query.txt.gz')
    transform.generate_queries_file(queries)  # query.txt.gz

    print('generate train.txt.gz')
    transform.generate_review_file('train', train_users, train_products, train_reviews)  # train.txt.gz
    print('generate test.txt.gz')
    transform.generate_review_file('test', test_users, test_products, test_reviews)  # test.txt.gz

    print('generate train_query_idx.txt.gz')
    transform.generate_query_idx_file('train', products, queries)  # train_query_idx.txt.gz
    print('generate test_query_idx.txt.gz')
    transform.generate_query_idx_file('test', products, queries)  # test_query_idx.txt.gz

    # build essential parameters for output_qrels_json_query

    train_user_product_map = {}
    test_user_product_map = {}
    for i in range(len(train_users)):
        if train_users[i] not in train_user_product_map:
            train_user_product_map[train_users[i]] = {train_products[i], }
        else:
            train_user_product_map[train_users[i]].add(train_products[i])

    for i in range(len(test_users)):
        if test_users[i] not in test_user_product_map:
            test_user_product_map[test_users[i]] = {test_products[i], }
        else:
            test_user_product_map[test_users[i]].add(test_products[i])

    # Output qrels files

    print('generate train qrels')
    transform.output_qrels_json_query(test_user_product_map, 'test')

    print('generate test qrels')
    transform.output_qrels_json_query(train_user_product_map, 'train')


if __name__ == '__main__':
    main()
