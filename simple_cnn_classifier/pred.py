#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, MaxPool2D
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation


# 10層CNNの構築
def cnn(input_shape, classes):
    # 入力層
    inputs = Input(shape=(input_shape[0], input_shape[1], 3))

    # 1層目
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 2層目
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 3層目
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 4層目
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 5、6層目
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 7、8層目
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # 9、10層目
    x = Dense(256, kernel_initializer='he_normal')(x)
    x = Dense(classes, kernel_initializer='he_normal')(x)
    outputs = Activation('softmax')(x)


    return Model(inputs=inputs, outputs=outputs)




def main():
    directory = 'img' # 画像が保存されているフォルダ
    df_test = pd.read_csv('test.csv') # テストデータの情報がかかれたDataFrame
    label_list = ['AMD', 'DR_DM', 'Gla', 'MH', 'Normal', 'RD', 'RP', 'RVO'] # ラベル名
    image_size = (224, 224) # 入力画像サイズ
    classes = len(label_list) # 分類クラス数


    # ネットワーク構築&学習済み重みの読み込み
    model = cnn(image_size, classes)
    model.load_weights('model_weights.h5')


    # 推論
    X = df_test['filename'].values
    y_true = list(map(lambda x: label_list.index(x), df_test['label'].values))
    y_pred = []
    for file in tqdm(X, desc='pred'):
        # 学習時と同じ条件になるように画像をリサイズ&変換
        img = Image.open(f'{directory}/{file}')
        img = img.resize(image_size, Image.LANCZOS)
        img = np.array(img, dtype=np.float32)
        img *= 1./255
        img = np.expand_dims(img, axis=0)

        y_pred.append(np.argmax(model.predict(img)[0]))


    # 評価
    print(classification_report(y_true, y_pred, target_names=label_list))


if __name__ == "__main__":
    main()
