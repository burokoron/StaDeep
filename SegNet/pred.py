#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPool2D, UpSampling2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
import cv2



# SegNet(エンコーダー8層、デコーダー8層)の構築
def cnn(input_shape, classes):
    # 入力画像サイズは32の倍数でなければならない
    assert input_shape[0]%32 == 0, 'Input size must be a multiple of 32.'
    assert input_shape[1]%32 == 0, 'Input size must be a multiple of 32.'

    # エンコーダー
    ## 入力層
    inputs = Input(shape=(input_shape[0], input_shape[1], 3))

    ## 1層目
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    ## 2層目
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    ## 3層目
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    ## 4層目
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    ## 5、6層目
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    ## 7、8層目
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # デコーダー
    ## 1層目
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ## 2、3層目
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ## 4層目
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ## 5層目
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ## 6層目
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ## 7、8層目
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(classes, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    outputs = Activation('softmax')(x)


    return Model(inputs=inputs, outputs=outputs)



def main():
    directory = 'CaDIS' # 画像が保存されているフォルダ
    df_test = pd.read_csv('test.csv') # テストデータの情報がかかれたDataFrame
    image_size = (224, 224) # 入力画像サイズ
    classes = 36 # 分類クラス数


    # ネットワーク構築
    model = cnn(image_size, classes)
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
