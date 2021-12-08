import pandas as pd
from keras.layers import Input, Dense, concatenate, Multiply
from keras.models import Model
from tensorflow.keras import initializers
from scipy.special import comb
import tensorflow as tf
from tensorflow import keras
import numpy as np

df = pd.read_csv(filepath_or_buffer="sample3.csv", encoding="ms932", names=('y', 'x1', 'x2', 'x3'))
df.drop(index=1, axis=0)

train_data=df[['x1','x2','x3']]
train_labels=df['y']

# 入力層
inputs_1 = Input(shape=(1,), dtype='float32')
inputs_2 = Input(shape=(1,), dtype='float32')
inputs_3 = Input(shape=(1,), dtype='float32')

# 隠れ層（sigmoid） Dense(ノード数, 活性化関数)(入力層)
hidden1_1 = Dense(1, name='Layer_1', activation='sigmoid')(inputs_1)
hidden1_2 = Dense(1, name='Layer_2', activation='sigmoid')(inputs_2)
hidden1_3 = Dense(1, name='Layer_3', activation='sigmoid')(inputs_3)

# 包除積分(掛け算)
multiply1 = Multiply()([hidden1_1,hidden1_2])
multiply2 = Multiply()([hidden1_1,hidden1_3])
multiply3 = Multiply()([hidden1_2,hidden1_3])
multiply4 = Multiply()([hidden1_1,hidden1_2,hidden1_3])

# 結合
hidden2_m = concatenate([hidden1_1, hidden1_2, hidden1_3, multiply1, multiply2, multiply3, multiply4])

# 出力
predictions = Dense(1,name='Output', kernel_regularizer=keras.regularizers.l2(0.1), activation='relu')(hidden2_m)

# モデル生成
model = Model(inputs=[inputs_1, inputs_2, inputs_3], outputs=predictions)
#model.summary()

# モデルの可視化
#keras.utils.plot_model(model, "my_model.png")

optimizer = tf.keras.optimizers.RMSprop(0.001)
 
model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['mae', 'mse'])

history=model.fit([train_data['x1'],train_data['x2'],train_data['x3']], 
                    train_labels, validation_split=0.25, epochs=200, batch_size=16, verbose=0)

# 目的変数を予測
Y_train_pred = model.predict([train_data['x1'],train_data['x2'],train_data['x3']])
#Y_test_pred = model.predict([test_data_use['RM'],test_data_use['PTRATIO'],test_data_use['LSTAT']])

# 決定係数
from sklearn.metrics import r2_score

print('r^2 train data: ', r2_score(train_labels, Y_train_pred))

for i in range(12): # 0,1,2,6-10は重みなし
    d = model.layers[i]
    #print(f"ノード番号：{i}")
    #print("重み,バイアス")
    print(d.get_weights())