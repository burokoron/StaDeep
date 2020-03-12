#!/usr/bin/env python
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from applications.efficientnet import EfficientNetB0
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam



# 学習曲線のグラフを描き保存する
def plot_history(history):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))

    # [左側] metricsについてのグラフ
    L_title = 'Accuracy_vs_Epoch'
    axL.plot(history.history['accuracy'])
    axL.plot(history.history['val_accuracy'])
    axL.grid(True)
    axL.set_title(L_title)
    axL.set_ylabel('accuracy')
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
    directory = 'img' # 画像が保存されているフォルダ
    df_train = pd.read_csv('train.csv') # 学習データの情報がかかれたDataFrame
    df_validation = pd.read_csv('val.csv') # 検証データの情報がかかれたDataFrame
    df_test = pd.read_csv('test.csv') # テストデータの情報がかかれたDataFrame
    label_list = ['AMD', 'DR_DM', 'Gla', 'MH', 'Normal', 'RD', 'RP', 'RVO'] # ラベル名
    image_size = (224, 224) # 入力画像サイズ
    classes = len(label_list) # 分類クラス数
    batch_size = 32 # バッチサイズ
    epochs = 300 # エポック数
    loss = 'categorical_crossentropy' # 損失関数
    optimizer = Adam(lr=0.00001, amsgrad=True) # 最適化関数
    metrics = 'accuracy' # 評価方法
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


    # データの数に合わせて損失の重みを調整
    weight_balanced = {}
    for i, label in enumerate(label_list):
        weight_balanced[i] = (df_train['label'] == label).sum()
    max_count = max(weight_balanced.values())
    for label in weight_balanced:
        weight_balanced[label] = max_count / weight_balanced[label]
    print(weight_balanced)


    # ジェネレータの生成
    ## 学習データのジェネレータ
    datagen = ImageDataGenerator(rescale=1./255, **aug_params)
    train_generator = datagen.flow_from_dataframe(
        dataframe=df_train, directory=directory,
        x_col='filename', y_col='label',
        target_size=image_size, class_mode='categorical',
        classes=label_list,
        batch_size=batch_size)
    step_size_train = train_generator.n // train_generator.batch_size
    ## 検証データのジェネレータ
    datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = datagen.flow_from_dataframe(
        dataframe=df_validation, directory=directory,
        x_col='filename', y_col='label',
        target_size=image_size, class_mode='categorical',
        classes=label_list,
        batch_size=batch_size)
    step_size_validation = validation_generator.n // validation_generator.batch_size


    # ネットワーク構築
    base_model = EfficientNetB0(include_top=False, weights='imagenet', pooling='avg',
                       input_shape=(image_size[0], image_size[1], 3),
                       backend=tf.keras.backend, layers=tf.keras.layers,
                       models=tf.keras.models, utils=tf.keras.utils)
    x = Dense(256, kernel_initializer='he_normal')(base_model.output)
    x = Dense(classes, kernel_initializer='he_normal')(x)
    outputs = Activation('softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=outputs)

    model.summary()
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])


    # 学習
    history = model.fit_generator(
        train_generator, steps_per_epoch=step_size_train,
        epochs=epochs, verbose=1, callbacks=[mc_cb, rl_cb, es_cb],
        validation_data=validation_generator,
        validation_steps=step_size_validation,
        class_weight=weight_balanced,
        workers=3)

    # 学習曲線の保存
    plot_history(history)


    # テストデータの評価
    ## 学習済み重みの読み込み
    model.load_weights('model_weights.h5')

    ## 推論
    X = df_test['filename'].values
    y_true = list(map(lambda x: label_list.index(x), df_test['label'].values))
    y_pred = []
    for file in tqdm(X, desc='pred'):
        # 学習時と同じ条件になるように画像をリサイズ&変換
        img = Image.open(f'{directory}/{file}')
        img = img.resize(image_size)
        img = np.array(img, dtype=np.float32)
        img *= 1./255
        img = np.expand_dims(img, axis=0)

        y_pred.append(np.argmax(model.predict(img)[0]))

    ## 評価
    print(classification_report(y_true, y_pred, target_names=label_list))


if __name__ == "__main__":
    main()
