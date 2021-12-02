import os
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras import Input
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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
    activation = 'relu'

    input = Input(shape=(28, 28, 1))

    x = Conv2D(3, (2,2), padding='same', name='conv1')(input)
    x = Activation(activation, name='act1')(x)
    x = MaxPooling2D((5,5), name='pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(3, name='dense4')(x)

    x = Activation(activation, name='act4')(x)
    x = Dense(2, name='dense6')(x)
    output = Activation('softmax', name='last_act')(x)

    model = Model(input, output)

    return model

model = model_functional_api()

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
                                 outputs=model.get_layer('act4').output)

intermediate_output = intermediate_layer_model.predict(x_train)
#Y追加
intermediate_output=np.insert(intermediate_output, 0, _y_train, axis=1)

import pandas as pd 
pd.DataFrame(intermediate_output).to_csv('sample3.csv', index=False, header=False)

import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TFメッセージ非表示


# フィルタ可視化
def filter_vi(model):
    vi_layer = []

    # 可視化対象レイヤー
    vi_layer.append(model.get_layer('conv1'))
    #vi_layer.append(model.get_layer('dense4'))
    #vi_layer.append(model.get_layer('dense6'))

    for i in range(len(vi_layer)):
        # レイヤーのフィルタ取得
        target_layer = vi_layer[i].get_weights()[0] #重み
        filter_num = target_layer.shape[3]

        # ウィンドウ名定義
        fig = plt.gcf()
        fig.canvas.set_window_title(vi_layer[i].name + " filter visualization")

        # 出力
        for j in range(filter_num):
            plt.subplots_adjust(wspace=0.4, hspace=0.8)
            plt.subplot(filter_num / 6 + 1, 6, j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(f'filter {j}')
            plt.imshow(target_layer[:, :, 0, j], cmap="gray")
        plt.show()

filter_vi(model)

# 重み表示
#d1 = model.layers[5]
#print(d1.get_weights())

d2 = model.layers[7]
print(d2.get_weights())