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
    input_data = np.array(input_data)
    get_feature = FeatureExtractor(len(data_list[0]), len(input_data[0]),
                                   order=config['order'], dt=config['dt'], minus=config['minus'],
                                   need_bias=config['need_bias'], other_items=config['other_items'])
    Slice.clear(config["clustering_th"])
    slice_data = []
    chp_list = []
    w = config['window_size']
    for data, input_val, chp in zip(data_list, input_data, gt_point):
        sample_weight = config['resampling_interval']
        data_sam, input_sam = resample(data, input_val, sample_weight)
        change_points, err_list = find_change_point(data_sam, input_sam, get_feature, w,
                                                    change_th=config['change_th'])
        change_points = np.array(change_points) * sample_weight
        print("ChP:\t", change_points)
        # plt.plot(np.arange(w, w + len(err_list)) * sample_weight, err_list, linewidth=3)
        # # plt.plot(np.arange(len(data[0])), data[0], linewidth=3)
        # for cp in chp:
        #     plt.axvline(x=cp, color='g', linestyle='--', linewidth=1.5)
        # for cp in change_points:
        #     plt.axvline(x=cp, color='r', linestyle='--', linewidth=1.5)
        # plt.show()
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


def get_config(json_path, evaluation: Evaluation):
    logging.basicConfig(level=logging.ERROR)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    default_config = {'dt': 0.01, 'total_time': 10, 'sigma_measure': 0.0, 'sigma_process': 0.0,
                      'random_seed': (time.time_ns() % (2**32)), 'order': 3, 'window_size': 10,
                      'clustering_method': 'fit', 'minus': False, 'need_bias': True, 'other_items': '',
                      'kernel': 'linear', 'svm_c': 1e6, 'class_weight': 1.0, 'need_reset': False,
                      'self_loop': False, "resampling_interval": 1, "truncation_size": 5,
                      "clustering_th": 1.5, "change_th": 0.1, "n_sample_ratio": 1.0}
    config = {}
    if json_path.isspace() or json_path == '':
        config = default_config
    else:
        with open(json_path) as f:
            json_file = json.load(f)
            evaluation.submit(gt_mode_num=len(json_file.get('automaton', {'mode': []})['mode']))
            json_config = json_file.get('config', {})
            for (key, val) in default_config.items():
                if key in json_config.keys():
                    config[key] = json_config.pop(key)
                else:
                    config[key] = val
            if len(json_config) != 0:
                raise Exception('Invalid parameter: ' + str(json_config))
            f.close()
    return config, get_hash_code(json_file, config)


def main(json_path: str, data_path='data', need_creat=None, need_plot=True):
    evaluation = Evaluation(json_path)
    config, hash_code = get_config(json_path, evaluation)
    HybridAutomata.LoopWarning = not config['self_loop']
    random_generator = np.random.default_rng(config['random_seed'])
    print('config: ')
    for key, value in config.items():
        print(f'\t{key}: {value}')

    if need_creat is None:
        need_creat = check_data_update(hash_code, data_path)
    if need_creat:
        print("Data being generated!")
        creat_data(json_path, data_path, config['dt'], config['total_time'], random_generator)
        save_hash_code(hash_code, data_path)

    random_generator = np.random.default_rng(config['random_seed'])
    random_generator = np.random.default_rng(random_generator.integers(2 ** 32))

    mode_list = []
    data = []
    input_list = []
    gt_list = []
    clean_data = []
    clean_input_list = []
    clean_gt_list = []
    clean_mode_list = []
    clean_data_path = "clean_data"

    test_num = 6
    gt_counter = test_num

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)
    if not os.path.isabs(clean_data_path):
        clean_data_path = os.path.join(current_dir, clean_data_path)
    for root, dirs, files in os.walk(data_path):
        print("Loading data!")
        for file in sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group())):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            npz_file = np.load(os.path.join(root, file))
            state_data_temp, mode_data_temp = npz_file['state'], npz_file['mode']
            change_point_list = npz_file.get('change_points', get_ture_chp(mode_data_temp))
            gt_list.append(change_point_list)
            if gt_counter == 0:
                print()
            gt_counter -= 1
            print("GT:\t", change_point_list.tolist())
            data.append(state_data_temp)
            mode_list.append(mode_data_temp)
            input_list.append(npz_file['input'])
    for root, dirs, files in os.walk(clean_data_path):
        for file in sorted(files, key=lambda x: int(re.search(r'(\d+)', x).group())):
            if re.search(r"(.)*\.npz", file) is None:
                continue
            npz_file = np.load(os.path.join(root, file))
            state_data_temp, mode_data_temp = npz_file['state'], npz_file['mode']
            change_point_list = npz_file.get('change_points', get_ture_chp(mode_data_temp))
            clean_gt_list.append(change_point_list)
            clean_data.append(state_data_temp)
            clean_mode_list.append(mode_data_temp)
            clean_input_list.append(npz_file['input'])


    print("Be running!")
    evaluation.submit(gt_chp=gt_list[test_num:])
    evaluation.submit(train_mode_list=mode_list[test_num:])
    evaluation.start()
    sys, slice_data = run(data[test_num:], input_list[test_num:], config,
                          evaluation, gt_list[test_num:], random_generator)
    print(f"mode number: {len(sys.mode_list)}")
    print("Start simulation")
    all_fit_mode, all_gt_mode = get_mode_list(slice_data, mode_list[test_num:])
    mode_map, mode_map_inv = max_bipartite_matching(all_fit_mode, all_gt_mode)

    data_test = data[:test_num]
    mode_list_test = mode_list[:test_num]
    input_list_test = input_list[:test_num]
    clean_data_test = clean_data[:test_num]
    clean_mode_list_test = clean_mode_list[:test_num]
    clean_input_list_test = clean_input_list[:test_num]
    init_state_test = get_init_state(clean_data_test, mode_map, mode_list_test, config['order'])
    fit_data_list, mode_data_list = [], []
    draw_index = 0  # If it is None, draw all the test data
    for data_item, mode_item, input_item, init_state, clean_data_item, clean_mode_item in zip(
        data_test, mode_list_test, clean_input_list_test, init_state_test, clean_data_test, clean_mode_list_test):
        
        fit_data = [clean_data_item[:, i] for i in range(config['order'])]
        mode_data = list(clean_mode_item[:config['order']])
        sys.reset(init_state, input_item[:, :config['order']])
        for i in range(config['order'], data_item.shape[1]):
            state, mode, switched = sys.next(input_item[:, i])
            fit_data.append(state)
            mode_data.append(mode_map_inv.get(mode, -mode))
        fit_data = np.array(fit_data)
        evaluation.submit(mode_num=len(sys.mode_list))
        fit_data_list.append(np.transpose(fit_data))
        mode_data_list.append(mode_data)
        
        if need_plot and (draw_index == 0 or draw_index is None):
            need_plot = not need_plot
            
            for var_idx in range(data_item.shape[0]):
                plt.figure(figsize=(12, 6))
                
                plt.plot(np.arange(len(data_item[var_idx])), data_item[var_idx], 
                        color='c', label='Noisy Data', alpha=0.7, linewidth=1)
                
                plt.plot(np.arange(len(clean_data_item[var_idx])), clean_data_item[var_idx], 
                        color='y', label='Clean Data', alpha=0.9, linewidth=2, linestyle='--')

                plt.plot(np.arange(fit_data.shape[0]), fit_data[:, var_idx], 
                        color='r', label='Fitted Data', alpha=0.8, linewidth=2)
                
                plt.title(f'Variable {var_idx + 1} Comparison')
                plt.xlabel('Time Step')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
            
            plt.figure(figsize=(12, 6))
            
            plt.plot(np.arange(len(mode_item)), mode_item, 
                    color='c', label='Noisy Mode', alpha=0.7, linewidth=1, marker='o', markersize=3)
            
            plt.plot(np.arange(len(clean_mode_item)), clean_mode_item, 
                    color='y', label='Clean Mode', alpha=0.9, linewidth=2, linestyle='--', marker='s', markersize=4)
            
            plt.plot(np.arange(len(mode_data)), mode_data, 
                    color='r', label='Fitted Mode', alpha=0.8, linewidth=2, marker='^', markersize=4)
            
            plt.title('Mode Comparison')
            plt.xlabel('Time Step')
            plt.ylabel('Mode')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yticks(sorted(set(list(mode_item) + list(clean_mode_item) + list(mode_data))))
            plt.show()
            
        if draw_index is not None:
            draw_index -= 1
            
    evaluation.submit(fit_mode=mode_data_list, fit_data=np.array(fit_data_list),
                      gt_mode=clean_mode_list_test, gt_data=clean_data_test, dt=config['dt'])
    return evaluation.calc()


if __name__ == "__main__":
    eval_log = main("./automata/ATVA/ball.json", need_creat=True)
    print("Evaluation log:")
    for key_, val_ in eval_log.items():
        print(f"{key_}: {val_}")
