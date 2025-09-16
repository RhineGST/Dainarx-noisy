import random
import numpy as np
import os
from math import *
import src.DE as DE
import matplotlib.pyplot as plt
from src.HybridAutomata import HybridAutomata
import json
import time


def creat_data(json_path: str, data_path: str, dT: float, times: float):
    r"""
    :param json_path: File path of automata.
    :param data_path: Data storage path.
    :param dT: Discrete time.
    :param times: Total sampling time.
    """
    np.random.seed(seed := time.time_ns() % (2**32))
    print(f"Random seed set to: {seed}")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(json_path):
        json_path = os.path.join(current_dir, json_path)
    if not os.path.isabs(data_path):
        data_path = os.path.join(current_dir, data_path)

    if not os.path.exists(data_path):
        os.mkdir(data_path)
    else:
        files = os.listdir(data_path)
        for file in files:
            os.remove(os.path.join(data_path, file))

    with open(json_path, 'r') as f:
        data = json.load(f)
        sys = HybridAutomata.from_json(data['automaton'])
        config = data.get('config', {})
        sigma_measure = config.get('sigma_measure', 0.02)
        sigma_process = config.get('sigma_process', 0.02)
        state_id = 0
        cnt = 0
        for init_state in data['init_state']:
            cnt += 1
            
            sys_noisy = HybridAutomata.from_json(data['automaton'])
            sys_clean = HybridAutomata.from_json(data['automaton'])
            
            state_data = []
            mode_data = []
            input_data = []
            change_points = [0]
            sys_noisy.reset(init_state)
            
            clean_state_data = []
            clean_mode_data = []
            clean_input_data = []
            clean_change_points = [0]
            sys_clean.reset(init_state)
            
            now = 0.
            idx = 0
            while now < times:
                now += dT
                idx += 1
                
                state, mode, switched = sys_noisy.next(dT, sigma_process)
                state_data.append(np.array(state) +
                                  np.array([random.normalvariate(0, sigma_measure) for _ in range(len(state))]))
                mode_data.append(mode)
                input_data.append(sys_noisy.getInput())
                if switched:
                    change_points.append(idx)
                
                clean_state, clean_mode, clean_switched = sys_clean.next(dT, 0)
                clean_state_data.append(np.array(clean_state))
                clean_mode_data.append(clean_mode)
                clean_input_data.append(sys_clean.getInput())
                if clean_switched:
                    clean_change_points.append(idx)
            
            change_points.append(idx)
            clean_change_points.append(idx)
            
            state_data = np.transpose(np.array(state_data))
            input_data = np.transpose(np.array(input_data))
            mode_data = np.array(mode_data)
            np.savez(os.path.join(data_path, "test_data" + str(state_id)),
                     state=state_data, mode=mode_data, input=input_data, change_points=change_points)
            
            clean_state_data = np.transpose(np.array(clean_state_data))
            clean_input_data = np.transpose(np.array(clean_input_data))
            clean_mode_data = np.array(clean_mode_data)
            np.savez(os.path.join(data_path, "clean_data" + str(state_id)),
                     state=clean_state_data, mode=clean_mode_data, input=clean_input_data, change_points=clean_change_points)
            
            state_id += 1


if __name__ == "__main__":
    creat_data('automata/1.json', 'data', 0.01, 10)