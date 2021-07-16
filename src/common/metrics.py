import numpy as np
import time
from collections import Iterable

best_metric = [0] * 3


def mrr(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(float(index + 1))
    else:
        return 0


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def display(epoch, epoch_num, loss, h_r, m_r_r, n_d_c_g,
            start_time, prepare_time=None, forward_time=None, step_time=None):
    print(
        "Running Epoch {:03d}/{:03d}".format(epoch + 1, epoch_num),
        "loss:{:.3f}".format(float(loss)),
        "Hr {:.3f}, Mrr {:.3f}, Ndcg {:.3f}".format(h_r, m_r_r, n_d_c_g),
        "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)),
        "prepare:", time.strftime("%H: %M: %S", time.gmtime(prepare_time)) if prepare_time is not None else "...",
        "forward:", time.strftime("%H: %M: %S", time.gmtime(forward_time)) if forward_time is not None else "...",
        "step:", time.strftime("%H: %M: %S", time.gmtime(step_time)) if step_time is not None else "...",
        flush=True)
    record((h_r, m_r_r, n_d_c_g))
    if epoch + 1 == epoch_num:
        print('-----------Best Result:-----------')
        print('Hr: {:.3f}, Mrr: {:.3f}, Ndcg: {:.3f}'.format(best_metric[0], best_metric[1], best_metric[2]))
        print('----------------------------------')


def record(metrics: Iterable):
    """
    :param metrics: 3 element tuple, HR MRR NDCG respectively
    :return:
    """
    for i, metric in enumerate(metrics):
        best_metric[i] = metric if metric > best_metric[i] else best_metric[i]
