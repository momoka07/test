# 重み指定なし
import os
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras import Input
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import initializers

# 学習データとテストデータを取得する。
(_x_train_val, _y_train_val), (_x_test, _y_test) = mnist.load_data()

# データのフィルタリング
b = np.where(_y_train_val < 2)[0]  # 訓練データから「0」「1」の全インデックスの取得
_x_train_val, _y_train_val = _x_train_val[b], _y_train_val[b]  # そのインデックス行を抽出（＝フィルタリング）
c = np.where(_y_test < 2)[0]   # テストデータから「0」「1」の全インデックスの取得
_x_test, _y_test = _x_test[c], _y_test[c]      # そのインデックス行を抽出（＝フィルタリング）

# 学習中の検証データがないので、train_test_split()を使って学習データ8割、検証データを2割に分割する。test_sizeが検証データの割合になっている。
_x_train, _x_val, _y_train, _y_val = train_test_split(_x_train_val, _y_train_val, test_size=0.2)

# 学習、検証、テストデータの前処理用関数。
def preprocess(data, label=False):
    if label: # 教師データはto_categorical()でone-hot-encodingする。       
        data = to_categorical(data)
    else:
        data = data.astype('float32') / 255 # 0-255 -> 0-1
        # Kerasの入力データの形式は(ミニバッチサイズ、横幅、縦幅、チャネル数)である必要があるので、reshape()を使って形式を変換する。
        # (sample, width, height) -> (sample, width, height, channel)
        data = data.reshape((-1, 28, 28, 1))

    return data

x_train = preprocess(_x_train)
x_val= preprocess(_x_val)
x_test = preprocess(_x_test)

y_train = preprocess(_y_train, label=True)
y_val = preprocess(_y_val, label=True)
y_test = preprocess(_y_test, label=True)

def model_functional_api():

    input = Input(shape=(28, 28, 1))
    
    # フィルタごとにConv
    x1 = Conv2D(1, (5,5), padding='same', name='conv1_1', activation='relu')(input)
    x2 = Conv2D(1, (5,5), padding='same', name='conv1_2', activation='relu')(input)
    x3 = Conv2D(1, (5,5), padding='same', name='conv1_3', activation='relu')(input)

    # フィルタごとにPooling
    x1 = MaxPooling2D((3,3), name='pool1_1')(x1)
    x2 = MaxPooling2D((3,3), name='pool1_2')(x2)
    x3 = MaxPooling2D((3,3), name='pool1_3')(x3)

    x1 = Flatten(name='flatten1_1')(x1)
    x2 = Flatten(name='flatten1_2')(x2)
    x3 = Flatten(name='flatten1_3')(x3)   

    x1 = Dense(1, name='dense1_1', activation='relu')(x1) 
    x2 = Dense(1, name='dense1_2', activation='relu')(x2)
    x3 = Dense(1, name='dense1_3', activation='relu')(x3)

    # 結合
    x = concatenate([x1,x2,x3], name='concat')

    output = Dense(2, name='dense6', activation='softmax')(x)

    model = Model(input, output)

    return model


model = model_functional_api()

model.summary()

"""
# 可視化
model.summary()
from tensorflow import keras
keras.utils.plot_model(model, 'my_model_with_shape_info.png', show_shapes=True)
"""

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

history=model.fit(x_train, y_train, validation_split=0.25, epochs=3, batch_size=128, verbose=1)

# 中間層の出力
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('concat').output)

intermediate_output = intermediate_layer_model.predict(x_train)
#Y追加
intermediate_output=np.insert(intermediate_output, 0, _y_train, axis=1)

import pandas as pd 
pd.DataFrame(intermediate_output).to_csv('sample1.csv', index=False, header=False)

import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TFメッセージ非表示


# フィルタ可視化
def filter_vi(model):
    vi_layer = []

    # 可視化対象レイヤー
    vi_layer.append(model.get_layer('conv1_1'))
    vi_layer.append(model.get_layer('conv1_2'))
    vi_layer.append(model.get_layer('conv1_3'))

    for i in range(len(vi_layer)):
        # レイヤーのフィルタ取得
        target_layer = vi_layer[i].get_weights()[0] #重み
        print(target_layer)
        print(target_layer.shape)
        filter_num = target_layer.shape[3]

        # ウィンドウ名定義
        fig = plt.gcf()
        fig.canvas.manager.set_window_title(vi_layer[i].name + " filter visualization")

        # 出力
        for j in range(filter_num):
            plt.subplots_adjust(wspace=0.4, hspace=0.8)
            plt.subplot(int(filter_num / 6 + 1), 6, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f'filter {j}')
            plt.imshow(target_layer[:, :, 0, j], cmap="gray")
        plt.show()

filter_vi(model)
