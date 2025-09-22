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

# def find_change_point(data: np.array, input_data: np.array, get_feature, w: int = 10, merge_th=None):
#     r"""
#     :param data: (N, M) Sample points for N variables.
#     :param input_data: Input of system.
#     :param get_feature: Feature extraction function.
#     :param w: Slide window size, default is 10.
#     :param merge_th: Change point merge threshold. The default value is w.
#     :return: change_points, err_data: The change points, and the error in each position of N variables.
#     """
#     change_points = []
#     error_datas = []
#     tail_len = 0
#     pos = 0
#     last = None
#     if merge_th is None:
#         merge_th = w

#     eps = get_feature.get_eps(data)
#     err_list = []

#     while pos + w < data.shape[1]:
#         feature, now_err, fit_order = get_feature(data[:, pos:(pos + w)], input_data[:, pos:(pos + w)])
#         p_sw = np.max(now_err[0])
#         err = (p_sw * 100000) / np.min(data[:, pos:(pos + w)][0])
#         err_list.append(err)
        
#         if last is not None:
#             if (abs(err) > 400) and tail_len == 0:
#                 change_points.append(pos + w - 1)
#                 tail_len = w
#             tail_len = max(tail_len - 1, 0)
#         last = fit_order
#         pos += 1

#     res = mergeChangePoints(change_points, merge_th)
#     res.append(data.shape[1])
#     res.insert(0, 0)

#     return res, err_list

def find_change_point(data: np.array, input_data: np.array, get_feature, w: int = 10, merge_th=11, 
                     rise_threshold: float = 0.3, min_rise_magnitude: float = 0.1,
                     debug_points: list = None, enable_tail_check: bool = True):
    r"""
    :param data: (N, M) Sample points for N variables.
    :param input_data: Input of system.
    :param get_feature: Feature extraction function.
    :param w: Slide window size, default is 10.
    :param merge_th: Change point merge threshold. The default value is w.
    :param rise_threshold:
    :param min_rise_magnitude: 
    :param debug_points: 需要调试的特定点位置列表，如 [22, 45, 67]
    :param enable_tail_check: 是否启用tail_len检查，如果False则忽略tail_len条件
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
    
    # 调试信息存储
    debug_info = {}
    if debug_points is not None:
        for point in debug_points:
            debug_info[point] = {
                'pos': None,
                'current_error': None,
                'prev_error': None,
                'prev_errors_list': None,
                'rise_ratio': None,
                'rise_magnitude': None,
                'condition1': None,  # rise_ratio > rise_threshold
                'condition2': None,  # rise_magnitude > min_rise_magnitude
                'condition3': None,  # rise_magnitude * rise_ratio > 5
                'condition4': None,  # tail_len == 0
                'condition5': None,  # len(prev_errors) >= 2
                'condition6': None,  # prev_errors[-1] > 1e-6
                'tail_len_value': None,
                'all_conditions_met': False,
                'would_be_detected': False,
                'actually_detected': False,  # 实际是否被检测到
                'reason_not_detected': None  # 如果没有被检测到的原因
            }

    while pos + w < data.shape[1]:
        feature, now_err, fit_order = get_feature(data[:, pos:(pos + w)], input_data[:, pos:(pos + w)])
        p_sw = np.max(now_err[0])
        current_error = (p_sw * 1) / np.min(data[:, pos:(pos + w)][0])
        err_list.append(current_error)
        
        # 检查是否为调试点
        current_time_point = pos + w - 1
        if debug_points is not None and current_time_point in debug_points:
            debug_info[current_time_point]['pos'] = pos
            debug_info[current_time_point]['current_error'] = current_error
            debug_info[current_time_point]['prev_error'] = prev_errors[-1] if len(prev_errors) > 0 else None
            debug_info[current_time_point]['prev_errors_list'] = prev_errors.copy()
            debug_info[current_time_point]['tail_len_value'] = tail_len
            debug_info[current_time_point]['condition5'] = len(prev_errors) >= 2
            debug_info[current_time_point]['condition6'] = prev_errors[-1] > 1e-6 if len(prev_errors) > 0 else False
        
        # 检查所有前置条件
        basic_conditions_met = len(prev_errors) >= 2 and prev_errors[-1] > 1e-6
        
        if basic_conditions_met:
            rise_ratio = (current_error - prev_errors[-1]) / prev_errors[-1]
            rise_magnitude = current_error - prev_errors[-1]
            
            # 记录调试信息
            if debug_points is not None and current_time_point in debug_points:
                debug_info[current_time_point]['rise_ratio'] = rise_ratio
                debug_info[current_time_point]['rise_magnitude'] = rise_magnitude
                debug_info[current_time_point]['condition1'] = rise_ratio > rise_threshold
                debug_info[current_time_point]['condition2'] = rise_magnitude > min_rise_magnitude
                debug_info[current_time_point]['condition3'] = rise_magnitude * rise_ratio > 5
                debug_info[current_time_point]['condition4'] = tail_len == 0
                
                # 计算所有条件是否满足
                detection_conditions = [
                    rise_ratio > rise_threshold,
                    rise_magnitude > min_rise_magnitude,
                    rise_magnitude * rise_ratio > 5
                ]
                
                if enable_tail_check:
                    detection_conditions.append(tail_len == 0)
                
                all_detection_conditions_met = all(detection_conditions)
                debug_info[current_time_point]['all_conditions_met'] = all_detection_conditions_met
                debug_info[current_time_point]['would_be_detected'] = all_detection_conditions_met
                    
            # 检测大的上升沿
            detection_conditions = [
                rise_ratio > rise_threshold,
                rise_magnitude > min_rise_magnitude,
                rise_magnitude * rise_ratio > 5
            ]
            
            if enable_tail_check:
                detection_conditions.append(tail_len == 0)
            
            if all(detection_conditions):
                change_point_pos = pos + w - 1
                change_points.append(change_point_pos)
                tail_len = w
                
                # 记录实际检测情况
                if debug_points is not None and current_time_point in debug_points:
                    debug_info[current_time_point]['actually_detected'] = True
        
        prev_errors.append(current_error)
        if len(prev_errors) > 3:
            prev_errors.pop(0)
        tail_len = max(tail_len - 1, 0)
        pos += 1

    change_points = sorted(set(change_points))
    res = change_points
    res.append(data.shape[1])
    res.insert(0, 0)
    
    # 分析未检测到的原因
    if debug_points is not None:
        for point, info in debug_info.items():
            if not info['actually_detected']:
                reasons = []
                if not info['condition5']:
                    reasons.append("prev_errors长度不足")
                elif not info['condition6']:
                    reasons.append("前一点误差太小")
                elif not info['condition1']:
                    reasons.append(f"上升比例不足: {info['rise_ratio']:.4f} <= {rise_threshold}")
                elif not info['condition2']:
                    reasons.append(f"上升幅度不足: {info['rise_magnitude']:.4f} <= {min_rise_magnitude}")
                elif not info['condition3']:
                    reasons.append(f"综合指标不足: {info['rise_magnitude'] * info['rise_ratio']:.4f} <= 5")
                elif not info['condition4'] and enable_tail_check:
                    reasons.append(f"tail_len不为0: {info['tail_len_value']} != 0")
                else:
                    reasons.append("未知原因（可能条件判断逻辑问题）")
                
                info['reason_not_detected'] = reasons
    
    # 输出调试信息
    if debug_points is not None and debug_info:
        print("\n=== 调试信息 ===")
        print(f"当前参数: rise_threshold={rise_threshold}, min_rise_magnitude={min_rise_magnitude}")
        print(f"tail_len检查: {'启用' if enable_tail_check else '禁用'}")
        print(f"最终检测到的切换点: {change_points}")
        
        for point, info in debug_info.items():
            print(f"  位置: {point}")
            print(f"  当前误差: {info['current_error']:.4f}")
            print(f"  前一点误差: {info['prev_error']:.4f}")
            print(f"  prev_errors长度: {len(info['prev_errors_list']) if info['prev_errors_list'] is not None else 'N/A'}")
            print(f"  上升比例: {info['rise_ratio']:.4f} (阈值: {rise_threshold}) {'✓' if info['condition1'] else '✗'}")
            print(f"  上升幅度: {info['rise_magnitude']:.4f} (阈值: {min_rise_magnitude}) {'✓' if info['condition2'] else '✗'}")
            print(f"  综合指标: {info['rise_magnitude'] * info['rise_ratio']:.4f} (阈值: 5) {'✓' if info['condition3'] else '✗'}")
            print(f"  tail_len: {info['tail_len_value']} (需要: 0) {'✓' if info['condition4'] else '✗'}")
            print(f"  实际检测到: {'是' if info['actually_detected'] else '否'}")
            
            if not info['actually_detected'] and info['reason_not_detected']:
                print(f"  未检测原因: {', '.join(info['reason_not_detected'])}")

    return res, err_list