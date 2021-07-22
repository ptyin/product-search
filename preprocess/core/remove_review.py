from tqdm import tqdm

def remove_review(df, word_dict):
    """ Remove test review data and remove duplicate reviews."""
    df = df.reset_index(drop=True)
    df_test = df[df['filter'] == 'Test'].reset_index(drop=True)
    review_text = []
    review_words = []
    review_train_set = set()

    review_test = set(repr(df_test['reviewText'][i]) for i in range(len(df_test)))

    # progress = range(len(df)
    progress = tqdm(range(len(df)), desc='removing reviews', total=len(df), leave=False, unit_scale=True)
    for i in progress:
        r = repr(df['reviewText'][i])
        if r not in review_train_set and r not in review_test:
            review_train_set.add(r)
            review_text.append(df['reviewText'][i])
            review_words.append([word_dict[w] for w in df['reviewText'][i]])
        else:
            review_text.append('[]')
            review_words.append('[]')
    df['reviewText'] = review_text
    df['reviewWords'] = review_words
    return df
