import numpy as np
import time


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


def display(epoch, epoch_num, loss, h_r, m_r_r, n_d_c_g, start_time):
    print(
        "Running Epoch {:03d}/{:03d}".format(epoch + 1, epoch_num),
        "loss:{:.3f}".format(float(loss)),
        "Hr {:.3f}, Mrr {:.3f}, Ndcg {:.3f}".format(h_r, m_r_r, n_d_c_g),
        "costs:", time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time)))

