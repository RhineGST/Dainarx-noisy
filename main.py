import json
import os
import re
import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from CreatData import creat_data
from src.DEConfig import FeatureExtractor
from src.utils import *

from src.CurveSlice import Slice, slice_curve
from src.ChangePoints import find_change_point
from src.Clustering import clustering
from src.GuardLearning import guard_learning
from src.BuildSystem import build_system, get_init_state
from src.Evaluation import Evaluation
from src.HybridAutomata import HybridAutomata

def resample(data_list, input_list, weight):
    return data_list[:, ::weight], input_list[:, ::weight]


def run(data_list, input_data, config, evaluation: Evaluation, gt_point, rng):
    get_feature = FeatureExtractor(len(data_list[0]), 0,
                                   order=config['order'], minus=config['minus'],
                                   need_bias=config['need_bias'], other_items=config['other_items'],
                                   fitting_method=config["fitting_method"])
    Slice.clear(config["clustering_th"])
    slice_data = []
    chp_list = []
    w = config['window_size']
    for data, input_val in zip(data_list, input_data):
        sample_weight = config['resampling_interval']
        change_points, err_list = find_change_point(data, input_val, get_feature, w,
                                                    change_th=config['change_th'], resample_interval=sample_weight)
        change_points = np.array(change_points)
        print("ChP:\t", change_points)
        plt.plot(np.arange(w * sample_weight, w * sample_weight + len(err_list)), err_list, linewidth=3)
        plt.plot(np.arange(len(data[0])), data[0], linewidth=3)
        for cp in change_points:
            plt.axvline(x=cp, color='r', linestyle='--', linewidth=1.5)
        plt.show()
        chp_list.append(change_points)
        slice_curve(slice_data, data, input_val, change_points, get_feature, config['truncation_size'])
    evaluation.submit(chp=chp_list)
    evaluation.recording_time("change_points")
    Slice.Method = config['clustering_method']
    Slice.fit_threshold(slice_data)
    #跳过特征为空的无效的分割
    slice_data = [
        d for d in slice_data
        if d.feature is not None
        and len(d.feature) > 0
        and d.fit_order is not None
        and len(d.fit_order) == len(d.feature)
    ]
    clustering(slice_data, config['self_loop'])
    clustering_checker = [(c.mode, c.isFront) for c in slice_data]
    clustering_front = []
    for (a, b) in clustering_checker:
        if b:
            clustering_front.append(a)
    print("clustering_front:", clustering_front)
    evaluation.recording_time("clustering")
    adj = guard_learning(slice_data, get_feature, config, rng)
    evaluation.recording_time("guard_learning")
    sys = build_system(slice_data, adj, get_feature)
    evaluation.stop("total")
    evaluation.submit(slice_data=slice_data)
    return sys, slice_data


def get_config(json_path):
    logging.basicConfig(level=logging.ERROR)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    default_config = {'path': None, 'dt': 0.01, 'order': 3, 'window_size': 10,
                      'clustering_method': 'fit', 'minus': False, 'need_bias': True, 'other_items': '',
                      'kernel': 'linear', 'svm_c': 1e6, 'class_weight': 1.0, 'need_reset': False,
                      'self_loop': False, "resampling_interval": 1, "truncation_size": 5,
                      "clustering_th": 1.5, "change_th": 0.1, "n_sample_ratio": 1.0,
                      "fitting_method": "tls", 'random_seed': (time.time_ns() % (2**32)),
                      "init_mode": []}
    config = {}
    if json_path.isspace() or json_path == '':
        config = default_config
    else:
        with open(json_path) as f:
            json_file = json.load(f)
            json_config = json_file.get('config', {})
            for (key, val) in default_config.items():
                if key in json_config.keys():
                    config[key] = json_config.pop(key)
                else:
                    config[key] = val
            if len(json_config) != 0:
                raise Exception('Invalid parameter: ' + str(json_config))
            f.close()
    return config


def main(json_path: str, need_plot=True):
    evaluation = Evaluation(json_path)
    config = get_config(json_path)
    HybridAutomata.LoopWarning = not config['self_loop']
    print('config: ')
    for key, value in config.items():
        print(f'\t{key}: {value}')

    random_generator = np.random.default_rng(config['random_seed'])
    random_generator = np.random.default_rng(random_generator.integers(2 ** 32))

    mode_list = config['init_mode']
    data = []

    test_num = 3
    gt_counter = test_num
    data_path = config['path']

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)
    data = mat2np(data_path)
    input_data = []
    for i in range(len(data)):
        input_data.append(np.zeros((0, len(data[i][0]))))


    print("Be running!")
    evaluation.start()
    sys, slice_data = run(data[test_num:], input_data, config,
                          evaluation, [], random_generator)
    print(f"mode number: {len(sys.mode_list)}")
    print("Start simulation")

    data_test = data[:test_num]
    mode_list_test = mode_list[:test_num]
    input_list_test = input_data[:test_num]
    init_state_test = get_init_state(data_test, config['init_mode'], config['order'])
    fit_data_list, mode_data_list = [], []
    draw_index = 0  # If it is None, draw all the test data
    for data_item, mode_item, input_item, init_state in zip(
        data_test, mode_list_test, input_list_test, init_state_test):
        
        fit_data = [data_item[:, i] for i in range(config['order'])]
        mode_data = [mode_item] * config['order']
        sys.reset(init_state, input_item[:, :config['order']])
        for i in range(config['order'], data_item.shape[1]):
            state, mode, switched = sys.next(input_item[:, i])
            fit_data.append(state)
            mode_data.append(mode)
        fit_data = np.array(fit_data)
        fit_data_list.append(np.transpose(fit_data))
        mode_data_list.append(mode_data)
        
        if need_plot and (draw_index == 0 or draw_index is None):
            need_plot = not need_plot
            
            for var_idx in range(data_item.shape[0]):
                plt.figure(figsize=(12, 6))
                
                plt.plot(np.arange(len(data_item[var_idx])), data_item[var_idx], 
                        color='c', label='Noisy Data', alpha=0.7, linewidth=1)

                plt.plot(np.arange(fit_data.shape[0]), fit_data[:, var_idx], 
                        color='r', label='Fitted Data', alpha=0.8, linewidth=2)
                
                plt.title(f'Variable {var_idx + 1} Comparison')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
            
            plt.figure(figsize=(12, 6))

            plt.plot(np.arange(len(mode_data)), mode_data, 
                    color='r', label='Fitted Mode', alpha=0.8, linewidth=2, marker='^', markersize=4)
            
            plt.title('Mode Comparison')
            plt.xlabel('Time Step')
            plt.ylabel('Mode')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yticks(sorted(set(list(mode_data))))
            plt.show()
            
        if draw_index is not None:
            draw_index -= 1
            
    evaluation.submit(fit_mode=mode_data_list, fit_data=fit_data_list, gt_data=data_test)
    return evaluation.calc()


if __name__ == "__main__":
    eval_log = main("./automata/ATVA/ball.json")
    print("Evaluation log:")
    for key_, val_ in eval_log.items():
        print(f"{key_}: {val_}")
