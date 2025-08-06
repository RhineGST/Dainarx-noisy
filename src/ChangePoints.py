import numpy as np
from src.DE import DE


def mergeChangePoints(data, th: float):
    data = np.unique(np.sort(data))
    res = []
    last = None
    for pos in data:
        if last is None or pos - last > th:
            res.append(pos)
        last = pos
    return res


def find_change_point(data: np.array, input_data: np.array, get_feature, w: int = 10, merge_th=None):
    r"""
    :param data: (N, M) Sample points for N variables.
    :param input_data: Input of system.
    :param get_feature: Feature extraction function.
    :param w: Slide window size, default is 10.
    :param merge_th: Change point merge threshold. The default value is w.
    :return: change_points, err_data: The change points, and the error in each position of N variables.
    """
    change_points = []
    error_datas = []
    tail_len = 0
    pos = 0
    last = None
    if merge_th is None:
        merge_th = w

    eps = get_feature.get_eps(data)
    err_list = []

    while pos + w < data.shape[1]:
        feature, now_err, fit_order = get_feature(data[:, pos:(pos + w)], input_data[:, pos:(pos + w)])
        p_sw = np.max(now_err[0])
        err_list.append((p_sw * 1000) / np.min(data[:, pos:(pos + w)][0]))
        
        # if last is not None:
        #     if (max(np.abs(now_err)) > 0.18) and tail_len == 0:
        #         change_points.append(pos + w - 1)
        #         tail_len = w
        #     tail_len = max(tail_len - 1, 0)
        # last = fit_order
        pos += 1

    res = mergeChangePoints(change_points, merge_th)
    res.append(data.shape[1])
    res.insert(0, 0)

    return res, err_list
