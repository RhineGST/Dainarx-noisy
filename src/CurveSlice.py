import copy

import numpy as np
import warnings


class Slice:
    RelativeErrorThreshold = []
    AbsoluteErrorThreshold = []
    ToleranceRatio = 0.1
    FitErrorThreshold = 1.
    ClusteringThreshold = 1.5
    Method = 'fit'

    @staticmethod
    def clear(clustering_threshold=1.5):
        Slice.RelativeErrorThreshold = []
        Slice.AbsoluteErrorThreshold = []
        Slice.ToleranceRatio = 0.1
        Slice.FitErrorThreshold = 1.
        Slice.ClusteringThreshold = clustering_threshold
        Slice.Method = 'fit'

    @staticmethod
    def get_dis(v1, v2):
        dis = np.linalg.norm(v1 - v2, ord=1)
        d1 = np.linalg.norm(v1, ord=1)
        d2 = np.linalg.norm(v2, ord=1)
        d_min = min(d1, d2)
        relative_dis = dis / max(d_min, 1e-6)
        return relative_dis, dis

    @staticmethod
    def fit_threshold_one(get_feature, data1, data2):
        feature1 = data1.feature
        feature2 = data2.feature

        # 直接跳过特征为空的数据
        if feature1 is None or feature2 is None or len(feature1) == 0 or len(feature2) == 0:
            print(f"[Warning] Skip empty feature: "
                f"len(feature1)={0 if feature1 is None else len(feature1)}, "
                f"len(feature2)={0 if feature2 is None else len(feature2)}")
            return  # 跳过当前 pair

        if len(feature1) != len(feature2):
            print(f"[Warning] Skip inconsistent feature length: "
                f"len(feature1)={len(feature1)}, len(feature2)={len(feature2)}")
            return
        
        assert len(feature1) == len(feature2)
        _, err, fit_order = get_feature([data1.data, data2.data],
                                      [data1.input_data, data2.input_data], is_list=True)
        if fit_order <= max(data1.fit_order, data2.fit_order):
            Slice.FitErrorThreshold = min(Slice.FitErrorThreshold, max(err) * Slice.ToleranceRatio)
            Slice.FitErrorThreshold = max(Slice.FitErrorThreshold, 1e-6)
        while len(Slice.RelativeErrorThreshold) < len(feature1):
            Slice.RelativeErrorThreshold.append(1e-1)
            Slice.AbsoluteErrorThreshold.append(1e-1)
        idx = 0
        for v1, v2 in zip(feature1, feature2):
            relative_dis, dis = Slice.get_dis(v1, v2)
            if relative_dis > 1e-4:
                Slice.RelativeErrorThreshold[idx] = \
                    min(Slice.RelativeErrorThreshold[idx], relative_dis * Slice.ToleranceRatio)
            if dis > 1e-4:
                Slice.AbsoluteErrorThreshold[idx] = \
                    min(Slice.AbsoluteErrorThreshold[idx], max(dis * Slice.ToleranceRatio, 1e-6))
            idx += 1
        return True

    @staticmethod
    def fit_threshold(data: list):
        for i in range(len(data)):
            if data[i].isFront:
                continue
            Slice.fit_threshold_one(data[i].get_feature, data[i], data[i - 1])
        for s in data:
            s.check_valid()

    def check_valid(self):
        if self.valid and self.err > Slice.FitErrorThreshold:
            warnings.warn("Find a invalid segmentation!")
            self.valid = True # False

    def undo_truncation(self):
        (self.data, self.input_data) = self.ori_data

    def truncation(self):
        self.undo_truncation()
        self.data = self.data[:, self.truncation_size:-self.truncation_size]
        self.input_data = self.input_data[:, self.truncation_size:-self.truncation_size]

    def __init__(self, data, input_data, get_feature, isFront, length, truncation_size):
        self.truncation_size = truncation_size
        self.ori_data = (data, input_data)
        self.data = data[:, self.truncation_size:-self.truncation_size]
        self.input_data = input_data[:, self.truncation_size:-self.truncation_size]
        self.get_feature = get_feature
        self.valid = True
        if len(self.data[0]) > get_feature.order:
            self.feature, err, self.fit_order = get_feature(self.data, self.input_data)
            self.err = np.max(err)
        else:
            self.feature, self.err, self.fit_order = [], 1e6, []
        if not self.valid:
            warnings.warn("warning: find a invalid segmentation!")
        self.mode = None
        self.isFront = isFront
        self.idx = None
        self.length = length

    def setMode(self, mode):
        self.mode = mode

    def test_set(self, other_list):
        data, input_data, other_fit_order = [], [], None
        max_err = self.err
        for s in other_list:
            data.append(s.data)
            input_data.append(s.input_data)
            if other_fit_order is None:
                other_fit_order = copy.copy(s.fit_order)
            else:
                other_fit_order = min(other_fit_order, s.fit_order)
            max_err = max(max_err, s.err)

        _, err, fit_order = self.get_feature([self.data] + data, [self.input_data] + input_data, is_list=True)
        order_condition = True
        for i in range(len(fit_order)):
            order_condition = order_condition and fit_order[i] <= max(self.fit_order[i], other_fit_order[i])
        # return order_condition and max(err) < Slice.FitErrorThreshold
        return max(err) < max_err * Slice.ClusteringThreshold

    def __and__(self, other):
        if Slice.Method == 'dis':
            idx = 0
            for v1, v2 in zip(self.feature, other.feature):
                relative_dis, dis = Slice.get_dis(v1, v2)
                if relative_dis > Slice.RelativeErrorThreshold[idx] and \
                        dis > Slice.AbsoluteErrorThreshold[idx]:
                    return False
                idx += 1
            return True
        else:
            _, err, fit_order = self.get_feature([self.data, other.data],
                                               [self.input_data, other.input_data], is_list=True)
            order_condition = True
            for i in range(len(fit_order)):
                order_condition = order_condition and fit_order[i] <= max(self.fit_order[i], other.fit_order[i])
            return order_condition and max(err) < max(self.err, other.err) * 1.5# Slice.FitErrorThreshold


def slice_curve(cut_data, data, input_data, change_points, get_feature, truncation_size):
    last = 0
    for point in change_points:
        if point == 0:
            continue
        cut_data.append(Slice(data[:, last:point], input_data[:, last:point],
                              get_feature, last == 0, point - last, truncation_size))
        last = point
    return cut_data
