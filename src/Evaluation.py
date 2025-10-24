import numpy as np
import bisect
import time

from numba import njit

from src.utils import get_ture_chp


class Evaluation:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.last_time = None
        self.time_list = []
        self.data = {}

    def start(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def stop(self, name):
        self.time_list.append((name, time.time() - self.start_time))

    def recording_time(self, name):
        now = time.time()
        self.time_list.append((name, now - self.last_time))
        print(f"recording {name} time: {now - self.last_time}")
        self.last_time = now

    def submit(self, **args):
        self.data.update(args)

    def calc(self):
        dt = self.data['dt']
        res = {"name": self.name, 'train_tc': 0., 'clustering_error': None, 'tc': 0., 'max_diff': 0., 'mean_diff': 0.}
        # fit_mode = get_ture_chp(self.data['fit_mode'])
        # fit_data = self.data['fit_data']
        # gt_mode = get_ture_chp(self.data['gt_mode'])
        # gt_data = self.data['gt_data']
        for fit_mode, fit_data, gt_mode, gt_data in zip(self.data['fit_mode'], self.data['fit_data'],
                                                        self.data['gt_mode'], self.data['gt_data']):
            fit_mode = get_ture_chp(fit_mode)
            gt_mode = get_ture_chp(gt_mode)
            diff = np.abs(fit_data - gt_data)

            for var_idx in range(diff.shape[0]):
                diff[var_idx] /= np.max(np.abs(gt_data[var_idx]))

            res["tc"] = max(res["tc"], max_min_abs_diff(fit_mode, gt_mode) * dt, max_min_abs_diff(gt_mode, fit_mode) * dt)

            train_tc = 0.0
            for chp, gt in zip(self.data["chp"], self.data["gt_chp"]):
                train_tc = max(train_tc, max_min_abs_diff(chp, gt) * dt, max_min_abs_diff(gt, chp) * dt)
            res["train_tc"] = max(res["train_tc"], train_tc)
            res["max_diff"] = max(np.max(diff), res["max_diff"])
            res["mean_diff"] = res['mean_diff'] + np.mean(diff)
        res["mean_diff"] = res["mean_diff"] / len(self.data['fit_mode'])
        res["robust_metric"] = call_mean_diff(self.data['fit_data'], self.data['gt_data'])
        res["clustering_error"] = abs(self.data['gt_mode_num'] - self.data['mode_num'])
        res["time"] = self.time_list.copy()
        return res

def call_mean_diff(fit_data_list, gt_data_list):
    fit_data_list, gt_data_list = normalize_cur(fit_data_list, gt_data_list)
    diff, len_sum, cnt, cnt_pre = 0., 0., 0., 0.
    for fit_data, gt_data in zip(fit_data_list, gt_data_list):
        # plt.plot(np.arange(len(fit_data[0])), fit_data[0])
        # plt.plot(np.arange(len(gt_data[0])), gt_data[0])
        # plt.show()
        diff += dtw_l1(fit_data, gt_data)
        cnt += gt_data.shape[1]
        cnt_pre += fit_data.shape[1]
        len_sum += np.sum(np.sqrt(np.sum(gt_data ** 2, axis=0)))
    return (diff / cnt_pre) / (len_sum / cnt)

def normalize_cur(fit_data : np.array, gt_data : np.array):
    max_val = np.max(np.max(gt_data, axis=2), axis=0)
    min_val = np.min(np.min(gt_data, axis=2), axis=0)
    gap = max_val - min_val
    for trace_idx in range(len(fit_data)):
        for i in range(fit_data[trace_idx].shape[1]):
            fit_data[trace_idx][:, i] = (fit_data[trace_idx][:, i] - min_val) / gap
    for trace_idx in range(len(gt_data)):
        for i in range(gt_data[trace_idx].shape[1]):
            gt_data[trace_idx][:, i] = (gt_data[trace_idx][:, i] - min_val) / gap
    return fit_data, gt_data

@njit(fastmath=True)
def dtw_l1(x, y):
    n, m = x.shape[1], y.shape[1]
    dis = np.full((n + 1, m + 1), np.inf)
    dis[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(x[:, i - 1] - y[:, j - 1])
            dis[i, j] = cost + min(dis[i - 1, j], dis[i, j - 1], dis[i - 1, j - 1])
    return dis[n, m]

def max_min_abs_diff(a, b):
    sorted_b = sorted(b)
    max_diff = 0
    for x in a:
        pos = bisect.bisect_left(sorted_b, x)
        if pos == 0:
            diff = abs(sorted_b[0] - x)
        elif pos == len(sorted_b):
            diff = abs(sorted_b[-1] - x)
        else:
            left = sorted_b[pos - 1]
            right = sorted_b[pos]
            diff = min(abs(x - left), abs(x - right))
        max_diff = max(max_diff, diff)

    return max_diff


def eva_trace(mode, trace, gt_mode, gt_trace, Ts):
    tc = max(max_min_abs_diff(mode, gt_mode), max_min_abs_diff(gt_mode, mode))
    mean_diff = np.mean(np.abs(trace - gt_trace))
    return tc * Ts, mean_diff



