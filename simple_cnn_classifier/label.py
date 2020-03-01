#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import pandas as pd


# 広角眼底データセットのcsvファイルを読み込む
df = pd.read_csv('data.csv')

dataset = defaultdict(list)

for i in range(len(df)):
    # 付いているラベルを文字化する
    labels = ''
    if df.iloc[i]['AMD'] == 1:
        labels += '_AMD'
    if df.iloc[i]['RVO'] == 1:
        labels += '_RVO'
    if df.iloc[i]['Gla'] == 1:
        labels += '_Gla'
    if df.iloc[i]['MH'] == 1:
        labels += '_MH'
    if df.iloc[i]['DR'] == 1:
        labels += '_DR'
    if df.iloc[i]['RD'] == 1:
        labels += '_RD'
    if df.iloc[i]['RP'] == 1:
        labels += '_RP'
    if df.iloc[i]['AO'] == 1:
        labels += '_AO'
    if df.iloc[i]['DM'] == 1:
        labels += '_DM'
    if labels == '':
        labels = 'Normal'
    else:
        labels = labels[1:]

    # マルチラベルでない(DR+DMは除く)画像および
    # 数の少ないDR、DMおよび
    # ラベルが重複するがDR+DMより数の少ないDM意外の画像を抽出する
    if '_' not in labels or labels == 'DR_DM':
        if labels not in ('DR', 'AO', 'DM'):
            dataset['filename'].append(df.iloc[i]['filename'])
            dataset['id'].append(df.iloc[i]['filename'].split('_')[0].split('.')[0])
            dataset['label'].append(labels)

# csvファイルで保存
dataset = pd.DataFrame(dataset)
dataset.to_csv('dataset.csv', index=False)
