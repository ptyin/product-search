import os
import time
import torch
from torch.utils.data import DataLoader
from common import testing_progress
from common.metrics import display


def get_user_to_num(full_df):

    user_bought_num = full_df.groupby('userID').size().tolist()
    return user_bought_num


def test(model, full_df, test_df, item_map, candidates, evaluate_neg, config):
    user_to_num = get_user_to_num(full_df)
    clipped_user = [0] * 10

    def apply(user):
        num = user_to_num[user] // 2
        if num < len(clipped_user) and clipped_user[num] < 1000:
            clipped_user[num] += 1
            return num
        else:
            return -1
            # test_df['user_bought_num'] = test_df['userID'].map(lambda user: user_to_num[user])

    # test_df['user_bought_num'] = test_df['userID'].map(lambda user: user_to_num[user])
    test_df['group'] = test_df['userID'].map(apply)
    print(clipped_user)
    test_df_by_num_list = [test_df[test_df['group'] == num] for num in range(1, 6)]
    candidates = {item_map[asin]: [item_map[candidate] for candidate in one_candidates]
                  for asin, one_candidates in candidates.items()}
    save_path = os.path.join(config.save_path, '{}.pt'.format(config.save_str))
    if not os.path.exists(save_path):
        checkpoints = os.listdir(config.save_path)
        latest = -1
        for checkpoint in checkpoints:
            save_str, stamp = checkpoint.split('-')
            if save_str == config.save_str and int(stamp) > latest:
                latest = int(stamp)
        if latest != -1:
            latest_checkpoint = '{}-{}.pt'.format(config.save_str, latest)
        else:
            raise FileNotFoundError
        save_path = os.path.join(config.save_path, latest_checkpoint)
    model.load_state_dict(torch.load(save_path))
    for test_df_by_num in test_df_by_num_list:
        test_dataset = yield test_df_by_num
        yield
        test_loader = DataLoader(test_dataset, drop_last=True, batch_size=1, shuffle=False, num_workers=0)
        Mrr, Hr, Ndcg = evaluate_neg(model, test_dataset, testing_progress(test_loader, 0, 1, config.debug),
                                     candidates, 10)
        display(0, 0, 0, Hr, Mrr, Ndcg, time.time())
