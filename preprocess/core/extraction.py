import gzip
import itertools
from core import text_process
from tqdm import tqdm


def extraction(meta_path):
    """
    Extract useful information (i.e., categories, related)
    :param meta_path:
    :return:
    """
    length = len(gzip.open(meta_path, 'rb').readlines())
    with gzip.open(meta_path, 'rb') as g:
        categories, also_viewed = {}, {}
        # progress = g
        progress = tqdm(g, desc='extracting meta', total=length, leave=False, unit_scale=True)
        for line in progress:
            line = eval(line)
            asin = line['asin']
            if 'category' in line:
                categories[asin] = [line['category']]
            elif 'categories' in line:
                categories[asin] = line['categories']
            else:
                raise Exception('category or categories tag not in metadata')
            # related = line['related'] if 'related' in line else None
            #
            # # fill the also_related dictionary TODO change to also_view and also_buy
            # also_viewed[asin] = []
            # relations = ['also_viewed', 'buy_after_viewing']
            # if related:
            #     also_viewed[asin] = [related[r] for r in relations if r in related]
            #     also_viewed[asin] = itertools.chain.from_iterable(also_viewed[asin])
    return categories, also_viewed


def generate_queries(review_df, stop_words, count, categories, default_query):
    """
    :param review_df:
    :param stop_words:
    :param count:
    :param categories:
    :param default_query:
    :return:
    """
    queries, reviews = [], []
    word_set = set()
    # word_dict = {}
    # progress = range(len(review_df))
    progress = tqdm(range(len(review_df)), desc='generating queries',
                    total=len(review_df), leave=False, unit_scale=True)
    for i in progress:
        asin = review_df['asin'][i]
        review = review_df['reviewText'][i]
        category = categories[asin] if asin in categories and len(categories[asin][0]) > 0 else [default_query]

        # process queries
        qs = map(text_process.remove_dup, map(text_process.remove_char, category))
        qs = [[w for w in q if w not in stop_words] for q in qs]

        for q in qs:
            for w in q:
                word_set.add(w)

        # process reviews
        review = text_process.remove_char(review)
        review = [w for w in review if w not in stop_words]

        queries.append(qs)
        reviews.append(review)

    review_df['query'] = queries  # write query result to dataframe

    # filtering words counts less than count
    reviews = text_process.filter_words(reviews, count)
    for review in reviews:
        for w in review:
            word_set.add(w)

    review_df['reviewText'] = reviews
    word_dict = dict(zip(word_set, range(1, len(word_set) + 1)))  # start from 1
    review_df['queryWords'] = [[[word_dict[w] for w in q] for q in qs] for qs in queries]

    return review_df, word_dict
