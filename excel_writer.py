#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/31
"""
import xlsxwriter

from root_dir import ROOT_DIR
from utils import *


def output():
    data_dir = os.path.join(ROOT_DIR, 'data')
    # mp3_dir = os.path.join(data_dir, 'mp3')
    mp3_dir = '/Users/wang/Desktop/Data/meipai_most_followers_mp3_15193'
    paths, names = traverse_dir_files(mp3_dir)

    res_file = os.path.join(data_dir, 'res.xlsx')

    label_list = [u'ID', u'时长', u'音量', u'音速', u'音乐', u'说话', u'杂音']

    workbook = xlsxwriter.Workbook(res_file)
    worksheet = workbook.add_worksheet()

    row, col = 0, 0
    for label in label_list:
        worksheet.write(row, col, label)
        col = col + 1

    row, col = 1, 0
    for name in names:
        dr = name.replace('.mp3', '').split('_')[-1]
        worksheet.write(row, col, name)
        worksheet.write(row, col + 1, dr)
        row = row + 1

    workbook.close()


if __name__ == '__main__':
    output()
