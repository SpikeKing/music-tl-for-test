#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/6/1
"""
from multiprocessing import Pool

import librosa
import numpy as np
import os

from root_dir import ROOT_DIR
from utils import mkdir_if_not_exist, traverse_dir_files


def get_feature(y, sr, dim=256):
    """
    计算音频的特征值

    :param y: 音频帧
    :param sr: 音频帧率
    :param dim: 音频特征长度
    :return: (32, sample_bin)
    """
    hop_length = len(y) / (dim + 2) / 64 * 64  # 频率距离需要对于64取模

    # 32维特征值
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)  # 13dim
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)  # 12dim
    rmse = librosa.feature.rmse(y=y, hop_length=hop_length)  # 1dim
    sp_ce = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)  # 1dim
    sp_cf = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)  # 1dim
    sp_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)  # 1dim
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)  # 1dim
    poly = librosa.feature.poly_features(y=y, sr=sr, hop_length=hop_length)  # 2dim

    all_features = np.vstack([mfcc, chroma, rmse, sp_ce, sp_cf, sp_bw, zcr, poly])[:, :dim]
    # print all_features.shape
    return all_features


def is_same_line(td_arr):
    """
    判断是否行值全相等
    :param td_arr: 二维矩阵
    :return: True含有值相等的行
    """
    row_max = np.max(td_arr, axis=1)
    row_min = np.min(td_arr, axis=1)

    same_row = (row_max == row_min)
    return np.any(same_row)


def check_error_features(features):
    """
    检查含有nan和全相等行等异常数据

    :param features: 二维特征矩阵
    :return: 是否含有错误数据
    """
    if np.isnan(features).sum() > 0:
        return True
    elif is_same_line(features):
        return True
    return False


def generate_augment(params):
    """
    音频增强
    :param params: 参数，[文件路径，音频ID，存储文件夹]
    :return: None
    """
    file_path, name_id = params

    folder = os.path.join(ROOT_DIR, 'data', 'npy_data')
    mkdir_if_not_exist(folder)

    try:
        saved_path = os.path.join(folder, name_id + '.npy')
        if os.path.exists(saved_path):
            print("[INFO] 文件 %s 存在" % name_id)
            return

        y_o, sr = librosa.load(file_path)
        y, _ = librosa.effects.trim(y_o, top_db=40)  # 去掉空白部分

        if not np.any(y):
            print('[Exception] 音频 %s 空' % name_id)
            return

        duration = len(y) / sr
        if duration < 4:  # 过滤小于3秒的音频
            print('[INFO] 音频 %s 过短: %0.4f' % (name_id, duration))
            return

        features = get_feature(y, sr)
        if check_error_features(features):
            print('[Exception] 音频 %s 错误' % name_id)
            return

        np.save(saved_path, features)  # 存储原文件的npy
    except Exception as e:
        print('[Exception] %s' % e)
        return

    print '[INFO] 音频ID ' + name_id
    return


def mp_preprocessor(n_process=41):
    raw_data = os.path.join(ROOT_DIR, 'data', 'raw_data')
    raw_dir = os.path.join(raw_data)
    paths, names = traverse_dir_files(raw_dir)
    p = Pool(processes=n_process)  # 进程数尽量与核数匹配
    print "[INFO] 数据数: %s" % len(paths)
    for path, name in zip(paths, names):
        name_id = name.replace('.mp3', '')
        params = (path, name_id)
        p.apply_async(generate_augment, args=(params,))
    p.close()
    p.join()


if __name__ == '__main__':
    mp_preprocessor()
