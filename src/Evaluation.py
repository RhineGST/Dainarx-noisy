import numpy as np
import bisect
import time
from numba import njit
from dtw import dtw
from src.utils import get_ture_chp
import matplotlib.pyplot as plt


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
        res = {"name": self.name, 'max_diff': 0., 'mean_diff': 0.}
        diff, len_sum, cnt, cnt_pre = 0., 0., 0., 0.
        for fit_data, gt_data in zip(self.data['fit_data'], self.data['gt_data']):
            fit_data, gt_data = normalize_cur(fit_data, gt_data)
            diff += dtw_l1(fit_data, gt_data)
            cnt += gt_data.shape[1]
            cnt_pre += fit_data.shape[1]
            # len_sum += np.linalg.norm(gt_data, axis=0).sum()
            len_sum += np.sum(np.sqrt(np.sum(gt_data ** 2, axis=0)))
            res["max_diff"] = max(np.max(diff), res["max_diff"])
        res["mean_diff"] = (diff / cnt_pre) / (len_sum / cnt)
        res["time"] = self.time_list.copy()
        return res

def call_mean_diff(fit_data_list, gt_data_list):
    diff, len_sum, cnt, cnt_pre = 0., 0., 0., 0.
    for fit_data, gt_data in zip(fit_data_list, gt_data_list):
        fit_data, gt_data = normalize_cur(fit_data, gt_data)
        # plt.plot(np.arange(len(fit_data[0])), fit_data[0])
        # plt.plot(np.arange(len(gt_data[0])), gt_data[0])
        # plt.show()
        diff += dtw_l1(fit_data, gt_data)
        cnt += gt_data.shape[1]
        cnt_pre += fit_data.shape[1]
        len_sum += np.sum(np.sqrt(np.sum(gt_data ** 2, axis=0)))
    return (diff / cnt_pre) / (len_sum / cnt)

def normalize_cur(fit_data : np.array, gt_data : np.array):
    max_val = np.max(gt_data, axis=1)
    min_val = np.min(gt_data, axis=1)
    gap = max_val - min_val
    for i in range(fit_data.shape[1]):
        fit_data[:, i] = (fit_data[:, i] - min_val) / gap
    for i in range(gt_data.shape[1]):
        gt_data[:, i] = (gt_data[:, i] - min_val) / gap
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


def eva_trace(trace, gt_trace):
    mean_diff = np.mean(np.abs(trace - gt_trace))
    return mean_diff



