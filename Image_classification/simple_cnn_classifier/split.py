#!/usr/bin/env python
# -*- coding: utf-8 -*-

# データセットの交差検証

import dataclasses
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm


@dataclasses.dataclass
class StratifiedGroupKFfold:
    """
    データをグループ層化K分割するときのパラメータを保持する
    """
    n_splits: int = 5 # 分割数
    shuffle: bool = False # シャッフルするかどうか
    random_state: int = None # ランダムシード


    def split(self, X, y, groups=None):
        """
        グループ層化K分割する

        Parameters
        ----------
        X : array-like, shape(ファイル数,)
            分割するファイル名
        y : array-like, shape(ファイル数,)
            分割するファイル名のラベル
        groups : None or array-like, shape(ファイル数,)
            分割するファイルのグループ名
            Noneの場合はただの層化K分割となる

        Yields
        -------
        train_index : array-like, shape(分割数, ファイル数)
            学習用として分けられたi分割目のXのインデックス
        test_index : array-like, shape(分割数, ファイル数)
            テスト用として分けられたi分割目のXのインデックス
        """

        # 初期化
        ## グループがない場合はファイル名をグループ名とする
        ## ユニークなグループ名を取得
        if groups is None:
            groups = X
        unique_group_list = list(set(groups))

        ## ラベルの数と種類を取得
        labels_list = list(set(y))
        labels_num = len(labels_list)
        y_count = np.zeros(labels_num)
        for _y in y:
            y_count[labels_list.index(_y)] += 1

        ## グループとファイル名の対応辞書，ファイル名とラベルの対応辞書，
        ## グループとラベルの数および種類の対応辞書を作成
        group_X_dict = defaultdict(list)
        X_y_dict = defaultdict(list)
        group_y_count_dict = defaultdict(lambda: np.zeros(labels_num))

        for _X, _y, _groups in zip(X, y, groups):
            group_X_dict[_groups].append(_X)
            idx = labels_list.index(_y)
            X_y_dict[_X] = idx
            group_y_count_dict[_groups][idx] += 1

        ## 分割後の情報を保存する変数の初期化
        group_X_fold = [[] for i in range(self.n_splits)]
        group_y_count_fold = [np.zeros(labels_num)
                              for i in range(self.n_splits)]

        # グループを1単位としてシャッフル
        if self.shuffle is True:
            np.random.seed(seed=self.random_state)
            np.random.shuffle(unique_group_list)

        # グループ層化K分割
        # 各分割群のラベル数を調べ，
        # ラベル数の標準偏差が最小になるようにデータを割り当てる
        for unique_group in tqdm(unique_group_list, desc='k-fold_split'):
            best_fold = None
            min_value = None
            for i in range(self.n_splits):
                group_y_count_fold[i] += group_y_count_dict[unique_group]
                std_per_label = []
                for label in range(labels_num):
                    label_std = np.std([group_y_count_fold[i][label]
                                        / y_count[label]
                                        for i in range(self.n_splits)])
                    std_per_label.append(label_std)
                group_y_count_fold[i] -= group_y_count_dict[unique_group]
                value = np.mean(std_per_label)

                if min_value is None or value < min_value:
                    min_value = value
                    best_fold = i

            group_y_count_fold[best_fold] += group_y_count_dict[unique_group]
            group_X_fold[best_fold] += group_X_dict[unique_group]

        # i番目の分割群をテストデータ，残りを学習データとする
        X_set = set(X)
        for i in range(self.n_splits):
            X_train = X_set - set(group_X_fold[i])
            X_test = set(group_X_fold[i])

            train_index = [i for i, _X in enumerate(X) if _X in X_train]
            test_index = [i for i, _X in enumerate(X) if _X in X_test]

            yield train_index, test_index



# csvファイルを読み込む
df = pd.read_csv('dataset.csv')

X = df['filename'].values
y = df['label'].values
groups = df['id'].values

# グループ層化K分割
df_train_list = []
df_val_list = []
df_test_list = []
kflod = StratifiedGroupKFfold(n_splits=5, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kflod.split(X, y, groups)):
    if i < 3:
        df_train_list += test_index
    elif i < 4:
        df_val_list += test_index
    else:
        df_test_list += test_index


## 分割されたデータの情報を出力
df_train = df.iloc[df_train_list]
df_val = df.iloc[df_val_list]
df_test = df.iloc[df_test_list]
df_train.to_csv(f'train.csv', index=False, encoding='utf-8')
df_val.to_csv(f'val.csv', index=False, encoding='utf-8')
df_test.to_csv(f'test.csv', index=False, encoding='utf-8')
