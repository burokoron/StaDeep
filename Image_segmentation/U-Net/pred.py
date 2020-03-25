#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPool2D, UpSampling2D, concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
import cv2



# U-Net(エンコーダー8層、デコーダー8層)の構築
@dataclasses.dataclass
class CNN:
    input_shape: tuple # 入力画像サイズ
    classes: int # 分類クラス数

    def __post_init__(self):
        # 入力画像サイズは32の倍数でなければならない
        assert self.input_shape[0]%32 == 0, 'Input size must be a multiple of 32.'
        assert self.input_shape[1]%32 == 0, 'Input size must be a multiple of 32.'


    # エンコーダーブロック
    @staticmethod
    def encoder(x, blocks, filters, pooling):
        for i in range(blocks):
            x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        if pooling:
            return MaxPool2D(pool_size=(2, 2))(x), x
        else:
            return x


    # デコーダーブロック
    @staticmethod
    def decoder(x1, x2, blocks, filters):
        x = UpSampling2D(size=(2, 2))(x1)
        x = concatenate([x, x2], axis=-1)

        for i in range(blocks):
            x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)

        return x


    def create(self):
        # エンコーダー
        inputs = Input(shape=(self.input_shape[0], self.input_shape[1], 3)) # 入力層
        x, x1 = self.encoder(inputs, blocks=1, filters=32, pooling=True) # 1層目
        x, x2 = self.encoder(x, blocks=1, filters=64, pooling=True) # 2層目
        x, x3 = self.encoder(x, blocks=1, filters=128, pooling=True) # 3層目
        x, x4 = self.encoder(x, blocks=1, filters=256, pooling=True) # 4層目
        x, x5 = self.encoder(x, blocks=2, filters=512, pooling=True) # 5、6層目
        x = self.encoder(x, blocks=2, filters=1024, pooling=False) # 7、8層目

        # デコーダー
        x = self.encoder(x, blocks=1, filters=1024, pooling=False) # 1層目
        x = self.decoder(x, x5, blocks=2, filters=512) # 2、3層目
        x = self.decoder(x, x4, blocks=1, filters=256) # 4層目
        x = self.decoder(x, x3, blocks=1, filters=128) # 5層目
        x = self.decoder(x, x2, blocks=1, filters=64) # 6層目
        ## 7、8層目
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([x, x1], axis=-1)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        outputs = Activation('softmax')(x)


        return Model(inputs=inputs, outputs=outputs)



def main():
    directory = 'CaDIS' # 画像が保存されているフォルダ
    df_test = pd.read_csv('test.csv') # テストデータの情報がかかれたDataFrame
    image_size = (224, 224) # 入力画像サイズ
    classes = 36 # 分類クラス数


    # ネットワーク構築
    model = CNN(input_shape=image_size, classes=classes).create()
    model.summary()
    model.load_weights('model_weights.h5')


    # 推論
    dict_iou = defaultdict(list)
    for i in tqdm(range(len(df_test)), desc='predict'):
        img = cv2.imread(f'{directory}/{df_test.at[i, "filename"]}')
        height, width = img.shape[:2]
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_LANCZOS4)
        img = np.array(img, dtype=np.float32)
        img *= 1./255
        img = np.expand_dims(img, axis=0)
        label = cv2.imread(f'{directory}/{df_test.at[i, "label"]}', cv2.IMREAD_GRAYSCALE)

        pred = model.predict(img)[0]
        pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_LANCZOS4)

        ## IoUの計算
        pred = np.argmax(pred, axis=2)
        for j in range(classes):
            y_pred = np.array(pred == j, dtype=np.int)
            y_true = np.array(label == j, dtype=np.int)
            tp = sum(sum(np.logical_and(y_pred, y_true)))
            other = sum(sum(np.logical_or(y_pred, y_true)))
            if other != 0:
                dict_iou[j].append(tp/other)

    # average IoU
    for i in range(classes):
        if i in dict_iou:
            dict_iou[i] = sum(dict_iou[i]) / len(dict_iou[i])
        else:
            dict_iou[i] = -1
    print('average IoU', dict_iou)



if __name__ == "__main__":
    main()
