import json
import os
import ast
import gzip


# ADH0O8UVJOT10

def main():
    # corpus = json.load(open('/disk/yxk/data/cold_start/meta_Musical_Instruments.json.gz'))
    with gzip.open('/disk/yxk/data/cold_start/reviews_Musical_Instruments_5.json.gz', 'rt') as fin:
        for line in fin:
            meta = ast.literal_eval(line)
            # if meta['reviewerID'] == 'ADH0O8UVJOT10':
            #     review = meta['reviewText']
            #     if review.find('rock') <= 0 < review.find('guitar'):
            #         print(review)
            #     print()

            if meta['reviewerID'] != 'ADH0O8UVJOT10' and \
                meta['asin'] in ['B000RW0O02', 'B000U0DU34', 'B000ZJTPLG',
                                 'B000RNB720', 'B005LYIW3W', 'B007Q28BHE',
                                 'B0002CZVBE', 'B0002F4VBM', 'B004U1QDL0',
                                 'B0002CZVK0', 'B005M0MUQK', 'B0043RZ9QQ',
                                 'B005F3H6Q8', 'B000T4PJC6', 'B005M0TKL8',
                                 'B004FEGXDK', 'B004OK1G64', 'B000PAPO9W']:
                review = meta['reviewText']
                if review.find('rock') > 0:
                    print(review)
                # print()


if __name__ == '__main__':
    main()
