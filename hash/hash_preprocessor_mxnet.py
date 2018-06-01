#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/8
"""
import os
import sys

import mxnet as mx
import numpy as np
from mxnet import gluon

from root_dir import ROOT_DIR

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)


class HashPreProcessor(object):
    def __init__(self):
        self.model = None
        self.data_dir = os.path.join(ROOT_DIR, 'data')
        self.gpu = False
        pass

    def process(self):
        print('[INFO] 转换开始')
        if self.gpu:
            ctx = mx.gpu(0)  # GPU模型
        else:
            ctx = mx.cpu(0)  # CPU模型

        sym = os.path.join(self.data_dir, "triplet_net.json")
        params = os.path.join(self.data_dir, "triplet_loss_model_88_0.9934.params")
        self.model = gluon.nn.SymbolBlock(outputs=mx.sym.load(sym), inputs=mx.sym.var('data'))
        self.model.load_params(params, ctx=ctx)

        file_name = 'data_all.npz'
        data_path = os.path.join(ROOT_DIR, 'data', file_name)
        data_all = np.load(data_path)
        X_test = data_all['f_list']
        l_list = data_all['l_list']
        n_list = data_all['n_list']

        print('[INFO] X_test.shape: ' + str(X_test.shape))

        print('[INFO] 转换数量: %s' % n_list.shape[0])

        X_test = np.transpose(X_test, [0, 2, 1])

        X_test = mx.nd.array(X_test).as_in_context(ctx)
        print('[INFO] 输入结构: %s' % str(X_test.shape))
        res = self.model(X_test)
        print('[INFO] 输出结构: %s' % str(res.shape))
        data = res.asnumpy()
        print('[INFO] data.shape: %s' % str(data.shape))
        oz_arr = np.where(data >= 0.5, 1.0, 0.0).astype(int)  # sigmoid激活函数
        print oz_arr[0]
        print np.sum(oz_arr, axis=1)  # 测试分布
        oz_bin = np.apply_along_axis(self.to_binary, axis=1, arr=oz_arr)
        print('[INFO] oz_bin: %s' % oz_bin[0])

        out_path = os.path.join(ROOT_DIR, 'data', 'data_all.bin.mx.npz')
        np.savez(out_path, b_list=oz_bin, l_list=l_list, n_list=n_list)

        print('[INFO] 输出示例: %s %s %s' % (str(oz_bin.shape), bin(oz_bin[0]), oz_bin[0]))
        print('[INFO] 转换结束')

    @staticmethod
    def to_binary(bit_list):
        out = long(0)  # 必须指定为long，否则存储过少
        # out = 0  # 必须指定为long，否则存储过少
        for bit in bit_list:
            out = (out << 1) | bit
        return out


if __name__ == '__main__':
    hpp = HashPreProcessor()
    hpp.process()
