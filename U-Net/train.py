#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MaxPool2D, UpSampling2D, concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import Sequence
import cv2



# ダイス係数
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / \
        (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return score


# ダイスロス
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)

    return loss


# カテゴリカルクロスエントロピー+ダイスロス
def cce_dice_loss(y_true, y_pred):
    loss = categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

    return loss


# データのジェネレータ
@dataclasses.dataclass
class TrainSequence(Sequence):
    directory: str # 画像が保存されているフォルダ
    df: pd.DataFrame # データの情報がかかれたDataFrame
    image_size: tuple # 入力画像サイズ
    classes: int # 分類クラス数
    batch_size: int # バッチサイズ
    aug_params: dict # ImageDataGenerator画像増幅のパラメータ

    def __post_init__(self):
        self.df_index = list(self.df.index)
        self.train_datagen = ImageDataGenerator(**self.aug_params)

    def __len__(self):
        return math.ceil(len(self.df_index) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.df_index[idx * self.batch_size:(idx+1) * self.batch_size]

        x = []
        y = []
        for i in batch_x:
            rand = np.random.randint(0, int(1e9))
            # 入力画像
            img = cv2.imread(f'{self.directory}/{self.df.at[i, "filename"]}')
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            img = np.array(img, dtype=np.float32)
            img = self.train_datagen.random_transform(img, seed=rand)
            img *= 1./255
            x.append(img)

            # セグメンテーション画像
            img = cv2.imread(f'{self.directory}/{self.df.at[i, "label"]}', cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)
            img = np.array(img, dtype=np.float32)
            img = np.reshape(img, (self.image_size[0], self.image_size[1], 1))
            img = self.train_datagen.random_transform(img, seed=rand)
            img = np.reshape(img, (self.image_size[0], self.image_size[1]))
            seg = []
            for label in range(self.classes):
                seg.append(img == label)
            seg = np.array(seg, np.float32)
            seg = seg.transpose(1, 2, 0)
            y.append(seg)

        x = np.array(x)
        y = np.array(y)


        return x, y


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


# 学習曲線のグラフを描き保存する
def plot_history(history):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))

    # [左側] metricsについてのグラフ
    L_title = 'Dice_coeff_vs_Epoch'
    axL.plot(history.history['dice_coeff'])
    axL.plot(history.history['val_dice_coeff'])
    axL.grid(True)
    axL.set_title(L_title)
    axL.set_ylabel('dice_coeff')
    axL.set_xlabel('epoch')
    axL.legend(['train', 'test'], loc='upper left')

    # [右側] lossについてのグラフ
    R_title = "Loss_vs_Epoch"
    axR.plot(history.history['loss'])
    axR.plot(history.history['val_loss'])
    axR.grid(True)
    axR.set_title(R_title)
    axR.set_ylabel('loss')
    axR.set_xlabel('epoch')
    axR.legend(['train', 'test'], loc='upper left')

    # グラフを画像として保存
    fig.savefig('history.jpg')
    plt.close()



def main():
    directory = 'CaDIS' # 画像が保存されているフォルダ
    df_train = pd.read_csv('train.csv') # 学習データの情報がかかれたDataFrame
    df_validation = pd.read_csv('val.csv') # 検証データの情報がかかれたDataFrame
    image_size = (224, 224) # 入力画像サイズ
    classes = 36 # 分類クラス数
    batch_size = 32 # バッチサイズ
    epochs = 300 # エポック数
    loss = cce_dice_loss # 損失関数
    optimizer = Adam(lr=0.001, amsgrad=True) # 最適化関数
    metrics = dice_coeff # 評価方法
    # ImageDataGenerator画像増幅のパラメータ
    aug_params = {'rotation_range': 5,
                  'width_shift_range': 0.05,
                  'height_shift_range': 0.05,
                  'shear_range': 0.1,
                  'zoom_range': 0.05,
                  'horizontal_flip': True,
                  'vertical_flip': True}


    # val_lossが最小になったときのみmodelを保存
    mc_cb = ModelCheckpoint('model_weights.h5',
                            monitor='val_loss', verbose=1,
                            save_best_only=True, mode='min')
    # 学習が停滞したとき、学習率を0.2倍に
    rl_cb = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3,
                              verbose=1, mode='auto',
                              min_delta=0.0001, cooldown=0, min_lr=0)
    # 学習が進まなくなったら、強制的に学習終了
    es_cb = EarlyStopping(monitor='loss', min_delta=0,
                          patience=5, verbose=1, mode='auto')


    # ジェネレータの生成
    ## 学習データのジェネレータ
    train_generator = TrainSequence(directory=directory, df=df_train,
                                    image_size=image_size, classes=classes,
                                    batch_size=batch_size, aug_params=aug_params)
    step_size_train = len(train_generator)
    ## 検証データのジェネレータ
    validation_generator = TrainSequence(directory=directory, df=df_validation,
                                         image_size=image_size, classes=classes,
                                         batch_size=batch_size, aug_params={})
    step_size_validation = len(validation_generator)


    # ネットワーク構築
    model = CNN(input_shape=image_size, classes=classes).create()
    model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])


    # 学習
    history = model.fit_generator(
        train_generator, steps_per_epoch=step_size_train,
        epochs=epochs, verbose=1, callbacks=[mc_cb, rl_cb, es_cb],
        validation_data=validation_generator,
        validation_steps=step_size_validation,
        workers=3)


    # 学習曲線の保存
    plot_history(history)


if __name__ == "__main__":
    main()
