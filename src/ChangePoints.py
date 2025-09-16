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


def find_change_point(data: np.array, input_data: np.array, get_feature, w: int = 10, merge_th=None, 
                     rise_threshold: float = 0.3, min_rise_magnitude: float = 5.0):
    r"""
    :param data: (N, M) Sample points for N variables.
    :param input_data: Input of system.
    :param get_feature: Feature extraction function.
    :param w: Slide window size, default is 10.
    :param merge_th: Change point merge threshold. The default value is w.
    :param rise_threshold:
    :param min_rise_magnitude: 
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
    prev_errors = []

    while pos + w < data.shape[1]:
        feature, now_err, fit_order = get_feature(data[:, pos:(pos + w)], input_data[:, pos:(pos + w)])
        p_sw = np.max(now_err[0])
        current_error = (p_sw * 1000) / np.min(data[:, pos:(pos + w)][0])
        err_list.append(current_error)
        
        if len(prev_errors) >= 2:
            # recent_gradient = np.mean(np.diff(prev_errors[-2:]))
            
            # 计算相对上升幅度（如果前一个点有效）
            if prev_errors[-1] > 1e-6:
                rise_ratio = (current_error - prev_errors[-1]) / prev_errors[-1]
                rise_magnitude = current_error - prev_errors[-1]
                        
                # 检测大的上升沿：上升比例大+上升幅度大
                if (
                    rise_ratio > rise_threshold and 
                    rise_magnitude > min_rise_magnitude and 
                    rise_magnitude * rise_ratio > 5 and
                    tail_len == 0):
                    
                    change_point_pos = pos + w - 1
                    change_points.append(change_point_pos)
                    tail_len = w
        
        prev_errors.append(current_error)
        if len(prev_errors) > 3:
            prev_errors.pop(0)
        tail_len = max(tail_len - 1, 0)
        pos += 1

    change_points = sorted(set(change_points))
    res = mergeChangePoints(change_points, merge_th)
    res.append(data.shape[1])
    res.insert(0, 0)

    return res, err_list
