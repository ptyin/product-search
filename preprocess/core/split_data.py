import pandas as pd
import numpy as np
from tqdm import tqdm


def split_data(df):
    """ Enlarge the dataset with the corresponding user-query-item pairs."""
    df_enlarge = {}
    i = 0
    # progress = range(len(df)
    progress = tqdm(range(len(df)), desc='enlarging', total=len(df), leave=False, unit_scale=True)
    for row in progress:
        for j in range(len(df['query'][row])):
            df_enlarge[i] = {'reviewerID': df['reviewerID'][row],
                             'userID': df['userID'][row], 'query': df['query'][row][j],
                             'queryWords': df['queryWords'][row][j],
                             'asin': df['asin'][row], 'reviewText': df['reviewText'][row],
                             'unixReviewTime': df['unixReviewTime'][row],
                             'reviewWords': None}
            i += 1
    df_enlarge = pd.DataFrame.from_dict(df_enlarge, orient='index')

    split_filter = []
    df_enlarge = df_enlarge.sort_values(by=['userID', 'unixReviewTime'])
    user_length = df_enlarge.groupby('userID').size().tolist()

    # progress = range(len(user_length)
    progress = tqdm(range(len(user_length)), desc='splitting', total=len(user_length), leave=False, unit_scale=True)
    for user in progress:
        length = user_length[user]
        # tag = ['Train' for _ in range(int(length * 0.7))]
        # tag_test = ['Test' for _ in range(length - int(length * 0.7))]
        tag = ['Train' for _ in range(length - 1)]
        tag_test = ['Test']
        tag.extend(tag_test)
        if length == 1:
            tag = ['Train']
        # np.random.shuffle(tag)
        split_filter.extend(tag)

    df_enlarge['filter'] = split_filter
    return df_enlarge.reset_index(drop=True)
