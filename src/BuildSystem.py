import copy

import numpy as np
from src.CurveSlice import Slice
from src.HybridAutomata import HybridAutomata
from src.DE import DE
from src.DE_System import DESystem
import matplotlib.pyplot as plt


class ModelFun:
    def __init__(self, model):
        self.model = copy.copy(model)

    def __call__(self, *args):
        return self.model.predict([[*args]])[0] > 0.5


def build_system(data: list[Slice], res_adj: dict, get_feature):
    data_of_mode = {}
    for cur in data:
        if not cur.valid:
            continue
        if data_of_mode.get(cur.mode) is None:
            data_of_mode[cur.mode] = [[], []]
        data_of_mode[cur.mode][0].append(cur.data)
        data_of_mode[cur.mode][1].append(cur.input_data)
    mode_list = {}
    for (mode, cur_list) in data_of_mode.items():
        raw_feature = get_feature(cur_list[0], cur_list[1], is_list=True)[0]
        X = np.array(raw_feature)
        X_mean = X.mean(axis=1)
        X_centered = X - X_mean
        if X_centered.shape[1] < 2 or np.allclose(X_centered, 0):
            print(f"[WARN] Data has no variance or too few samples: mode={mode}")
            X_reduced = np.zeros((X.shape[0], 1))
            continue
        print("SVDing")
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        total_variance = np.sum(S ** 2)
        print(f"S: {S}, total_variance: {total_variance}")
        explained_variance = np.cumsum(S ** 2) / total_variance
        rank = np.searchsorted(explained_variance, 0.95) + 1
        rank = min(rank, min(X.shape[0], X.shape[1], 10))
        X_reduced = np.dot(U[:, :rank], np.diag(S[:rank]))
        print(f"X_reduced: {X_reduced}, X: {X}, rank: {rank}")
        mode_list[mode] = DESystem(X_reduced, [], [], get_feature)
    adj = {}
    for (u, v), (model, reset_fun) in res_adj.items():
        if adj.get(u) is None:
            adj[u] = []
        adj[u].append((v, ModelFun(model), reset_fun))
    return HybridAutomata(mode_list, adj)


def get_init_state(data_list, mode_map, mode_list, bias):
    res = []
    for data, mode in zip(data_list, mode_list):
        if mode_map.get(mode[bias - 1]) is None:
            raise Exception("unknown mode: " + str(mode[bias - 1]))
        init_state = {'mode': mode_map[mode[bias - 1]]}
        for i in range(data.shape[0]):
            init_state['x' + str(i)] = data[i, (bias - 1)::-1]
        res.append(init_state)
    return res
