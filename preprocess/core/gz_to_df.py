import json
import gzip
import pandas as pd
from tqdm import tqdm


def get_df(path):
    """ Apply raw data to pandas DataFrame. """
    idx = 0
    df = {}
    length = len(gzip.open(path, 'rb').readlines())
    g = gzip.open(path, 'rb')
    # progress = g
    progress = tqdm(g, desc='transforming', total=length, leave=False, unit_scale=True)

    for line in progress:
        # if idx > 2000:
        #     break
        df[idx] = json.loads(line)
        idx += 1
    return pd.DataFrame.from_dict(df, orient='index')
