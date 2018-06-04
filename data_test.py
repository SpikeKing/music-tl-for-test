#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/6/4
"""
from datetime import datetime
import librosa
import os
import mxnet as mx
import numpy as np
import soundfile as sf
import io

from mxnet import gluon
from six.moves.urllib.request import urlopen

from data_preprocessor import get_feature
from root_dir import ROOT_DIR

path = os.path.join(ROOT_DIR, 'data', 'WeWillRockYou45-55.mp3')

y, sr = librosa.load(path)
features = get_feature(y, sr)


ctx = mx.cpu(0)
sym = os.path.join(ROOT_DIR, 'data', "triplet_net.json")
params = os.path.join(ROOT_DIR, 'data', "triplet_loss_model_88_0.9934.params")
model = gluon.nn.SymbolBlock(outputs=mx.sym.load(sym), inputs=mx.sym.var('data'))
model.load_params(params, ctx=ctx)

features = np.reshape(features, (1, 32, 256))
features = np.transpose(features, [0, 2, 1])
features = mx.nd.array(features).as_in_context(ctx)

start_time = datetime.now()  # 起始时间

for index in range(1000):
    features = get_feature(y, sr)

elapsed_time = (datetime.now() - start_time).total_seconds()  # 终止时间
print "[INFO] 耗时: %s (秒)" % elapsed_time
