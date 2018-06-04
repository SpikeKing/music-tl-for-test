#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/8
"""

import os
import sys
from datetime import datetime

import librosa
import mxnet as mx
import numpy as np
from mxnet import gluon

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from data_preprocessor import get_feature
from root_dir import ROOT_DIR
from utils import sort_two_list


class DistanceApi(object):
    def __init__(self):
        self.model = None
        self.data_dir = os.path.join(ROOT_DIR, 'data')

        self.b_list, self.l_list, self.n_list = self.load_data()

        pass

    def load_data(self):
        file_name = 'data_all.bin.mx.npz'
        print('模型名称: %s' % file_name)
        data_path = os.path.join(self.data_dir, file_name)
        data_all = np.load(data_path)
        b_list = data_all['b_list']
        l_list = data_all['l_list']
        n_list = data_all['n_list']
        return b_list, l_list, n_list

    def distance(self, audio_id):
        # 获取索引ID
        i_name = audio_id
        i_id = np.where(self.n_list == i_name)
        i_id = int(i_id[0])  # 索引ID

        print bin(self.b_list[i_id])

        def hamdist_for(data):  # Hamming距离
            return self.hamdist(self.b_list[i_id], data)

        start_time = datetime.now()  # 起始时间
        b_list_dist = [hamdist_for(x) for x in list(self.b_list)]
        elapsed_time = (datetime.now() - start_time).total_seconds()
        run_num = self.b_list.shape[0]
        tps = float(run_num) / float(elapsed_time)
        print "[INFO] Num: %s, Time: %s s, TPS: %0.0f (%s ms)" % (run_num, elapsed_time, tps, (1 / tps * 1000))

        sb_list, sn_list = sort_two_list(list(b_list_dist), list(self.n_list))
        return sb_list[0:20], sn_list[0:20]

    def init_mode(self):
        ctx = mx.cpu(0)

        sym = os.path.join(self.data_dir, "triplet_net.json")
        params = os.path.join(self.data_dir, "triplet_loss_model_88_0.9934.params")
        self.model = gluon.nn.SymbolBlock(outputs=mx.sym.load(sym), inputs=mx.sym.var('data'))
        self.model.load_params(params, ctx=ctx)

    def distance_for_mp3(self, mp3_path):
        ctx = mx.cpu(0)
        start_time = datetime.now()  # 起始时间
        input_b = None
        y_o, sr = librosa.load(mp3_path)
        y, _ = librosa.effects.trim(y_o, top_db=40)  # 去掉空白部分
        features = get_feature(y, sr)

        features = np.reshape(features, (1, 32, 256))
        features = np.transpose(features, [0, 2, 1])
        features = mx.nd.array(features).as_in_context(ctx)
        print('[INFO] 输入结构: %s' % str(features.shape))
        res = self.model(features)
        print('[INFO] 输出结构: %s' % str(res.shape))
        data_prop = res.asnumpy()
        oz_arr = np.where(data_prop >= 0.5, 1.0, 0.0).astype(int)
        input_b = self.to_binary(oz_arr[0])

        elapsed_time = (datetime.now() - start_time).total_seconds()
        tps = float(1.0) / float(elapsed_time)
        print "Time: %s s, TPS: %0.0f (%s ms)" % (elapsed_time, tps, (1 / tps * 1000))

        print bin(input_b)

        def hamdist_for(o_data):  # Hamming距离
            return self.hamdist(input_b, o_data)

        b_list_dist = [hamdist_for(x) for x in list(self.b_list)]
        sb_list, sn_list = sort_two_list(list(b_list_dist), list(self.n_list))
        return sb_list[0:20], sn_list[0:20]

    @staticmethod
    def hamdist(a, b):
        return bin(a ^ b).count('1')  # 更快

    @staticmethod
    def to_binary(bit_list):
        out = long(0)  # 必须指定为long，否则存储过少
        for bit in bit_list:
            out = (out << 1) | bit
        return out


def duplicates(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def test_of_acc():
    da = DistanceApi()  # 接口类
    label_list = np.unique(da.l_list)  # 标签类别
    label_indexes = [np.where(da.l_list == i)[0] for i in label_list]  # 标签索引列表

    n_all = 0
    n_error = 0
    for idx_list, label in zip(label_indexes, label_list):
        idx_first = idx_list[0].astype(int)  # 第1个索引值
        n_idx = len(idx_list)  # 索引数
        true_list = [da.n_list[idx] for idx in idx_list]  # 真实元素列表
        print('[INFO] ID: %s' % da.n_list[idx_first])
        print('[INFO] Num of IDs: %s' % n_idx)
        try:
            rb_list, rn_list = da.distance(da.n_list[idx_first])
        except:
            continue

        res_indexes = duplicates(rb_list, 0)  # duplicate indexes of same value 0
        res_list = [rn_list[idx] for idx in res_indexes]  # 最小值的元素

        sub_set = set(true_list) - set(res_list)

        n_all += n_idx

        rb_list, rn_list = sort_two_list(rb_list, rn_list)
        print '[INFO]',
        for rb, rn in zip(rb_list, rn_list):
            print '%s-%s ' % (rb, rn),
        print('')
        if sub_set:
            print(sub_set)
            n_error += len(sub_set)
        print('-' * 50)
    print('总数: %s' % n_all)
    print('错误: %s' % n_error)

    # label_num_dict = collections.defaultdict(int)
    # name_label_dict = dict()
    # for label, name in zip(da.l_list, da.n_list):
    #     label_num_dict[label] += 1
    #     name_label_dict[name] = label
    #
    # n_error = 0
    # n_all = 0
    # for label, name in zip(da.l_list, da.n_list):
    #     n_same = label_num_dict[label]
    #     rb_list, rn_list = da.distance(name)
    #     l_same = rn_list[slice(n_same)]
    #     for sl in l_same:
    #         if sl != label:
    #             n_error += 1  # 写入文档
    #         else:
    #             n_all += 1
    #
    # print('[INFO] 错误: %s' % n_error)
    # print('[INFO] 整体: %s' % n_error)
    # audio_name = '924643775'
    # print('[INFO] 目标音频: %s' % audio_name)
    # rb_list, rn_list = da.distance(audio_name)
    # print('[INFO] 距离: %s' % rb_list)
    # print('[INFO] 相似: %s' % rn_list)


def test_of_distance():
    da = DistanceApi()
    print(da.n_list)
    audio_name = '924643775'
    print('[INFO] 目标音频: %s' % audio_name)
    rb_list, rn_list = da.distance(audio_name)
    print('[INFO] 距离: %s' % rb_list)
    print('[INFO] 相似: %s' % rn_list)


def test_of_mp3():
    mp3_path = os.path.join(ROOT_DIR, 'experiments/raw_data/train', '993001815_15.05.mp3')
    da = DistanceApi()
    da.init_mode()
    print('[INFO] 目标音频: %s' % mp3_path)

    rb_list, rn_list = da.distance_for_mp3(mp3_path)
    print('[INFO] 距离: %s' % rb_list)
    print('[INFO] 相似: %s' % rn_list)


if __name__ == '__main__':
    test_of_acc()
