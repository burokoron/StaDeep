#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from collections import defaultdict
import pandas as pd


# 画像とラベルの対応を記述したcsvファイルを作成する
def make_csv(fpath, dirlist):
    # 学習画像のファイルパスを調べる
    dataset = defaultdict(list)
    for dir in dirlist:
        filelist = sorted(os.listdir(f'CaDIS/{dir}/Images'))
        dataset['filename'] += list(map(lambda x: f'{dir}/Images/{x}', filelist))
        filelist = sorted(os.listdir(f'CaDIS/{dir}/Labels'))
        dataset['label'] += list(map(lambda x: f'{dir}/Labels/{x}', filelist))

    # csvファイルで保存
    dataset = pd.DataFrame(dataset)
    dataset.to_csv(fpath, index=False)



# 学習データのビデオフォルダ
train_dir = ['Video01', 'Video03', 'Video04', 'Video06', 'Video08', 'Video09',
             'Video10', 'Video11', 'Video13', 'Video14', 'Video15', 'Video17',
             'Video18', 'Video20', 'Video21', 'Video22', 'Video23', 'Video24',
             'Video25']

# 検証データのビデオフォルダ
val_dir = ['Video05', 'Video07', 'Video16']

# テストデータのビデオフォルダ
test_dir = ['Video02', 'Video12', 'Video19']


# 学習データの画像とラベルの対応を記述したcsvファイルを作成する
make_csv('train.csv', train_dir)

# 検証データの画像とラベルの対応を記述したcsvファイルを作成する
make_csv('val.csv', val_dir)

# 学習データの画像とラベルの対応を記述したcsvファイルを作成する
make_csv('test.csv', test_dir)
