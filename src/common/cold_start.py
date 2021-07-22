import time
import torch
from torch.utils.data import DataLoader
from common import testing_progress
from common.metrics import display


def get_user_to_num(full_df):

    user_bought_num = full_df.groupby('userID').size().tolist()
    return user_bought_num


def test(model, full_df, test_df, item_map, candidates, evaluate_neg, config):
    # user_to_num = get_user_to_num(full_df)
    # test_df['user_bought_num'] = test_df['userID'].map(lambda user: user_to_num[user])
    test_df_by_num_list = [test_df[test_df['user_bought_num'] // 5 == num] for num in range(6)]
    candidates = {item_map[asin]: [item_map[candidate] for candidate in one_candidates]
                  for asin, one_candidates in candidates.items()}
    model.load_state_dict(torch.load(config.save_path))
    for test_df_by_num in test_df_by_num_list:
        test_dataset = yield test_df_by_num
        yield
        test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0)
        Mrr, Hr, Ndcg = evaluate_neg(model, test_dataset, testing_progress(test_loader, 0, 1, config.debug),
                                     candidates, 10)
        display(0, 0, 0, Hr, Mrr, Ndcg, time.time())
