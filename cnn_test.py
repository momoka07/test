import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

#MNISTデータセット呼び出し
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#データの正規化
X_train = np.array(X_train)/255.
X_test = np.array(X_test)/255.

#CNNに入れるために次元を追加
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

#ラベルのバイナリ化
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers.core import Flatten

inputs = Input(shape=(28,28,1)) #入力層

conv1 = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs) #畳込み層1
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(32, (3, 3), padding='same', activation='relu')(pool1) #畳込み層2
pool2 = MaxPooling2D((2, 2))(conv2)

flat = Flatten()(pool2) #全結合層に渡すため配列を一次元化
dense1 = Dense(784, activation='relu')(flat) #全結合層

predictions = Dense(10, activation='softmax')(dense1) #出力層

model = Model(inputs=inputs, outputs=predictions) #モデルの宣言(入力と出力を指定)

model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(X_train, y_train, batch_size=16, verbose=1, epochs=30, validation_split=0.3)

score = model.evaluate(X_test, y_test, verbose=1)
print("正解率(acc)：", score[1])