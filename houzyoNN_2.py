# 重みとバイアス初期値あり
import pandas as pd
from keras.layers import Input, Dense, concatenate, Multiply
from keras.models import Model
from tensorflow.keras import initializers
import tensorflow as tf

df = pd.read_csv(filepath_or_buffer="sample2.csv", encoding="ms932", names=('y', 'x1', 'x2', 'x3'))
df.drop(index=1, axis=0)

# データシャッフル
df_s = df.sample(frac=1)

column = ['x1','x2','x3']
train_data=df_s[column]
train_labels=df_s['y']

# 重みとバイアスの初期値を算出
max = train_data.max(axis=0) 
min = train_data.min(axis=0)
a = [] #重みの初期値を入れる用
b = [] #バイアスの初期値を入れる用

df_corr = df.corr() #相関係数計算

for num in column:
    if df_corr.loc['y',num] >= 0:# 相関係数が正
        a.append(6 / (max[num]-min[num]))
        b.append(-6 * (min[num] + max[num]) / (2 * (max[num]-min[num])))
    elif df_corr.loc['y',num] < 0:# 相関係数が負
        a.append(-6 / (max[num]-min[num]))
        b.append(6 * (min[num] + max[num]) / (2 * (max[num]-min[num])))

# 入力層
inputs_1 = Input(shape=(1,), dtype='float32')
inputs_2 = Input(shape=(1,), dtype='float32')
inputs_3 = Input(shape=(1,), dtype='float32')

# 隠れ層（sigmoid） Dense(ノード数, 活性化関数)(入力層)
hidden1_1 = Dense(1, name='Layer_1', activation='sigmoid',
                    kernel_initializer=initializers.Constant(value=a[1]),
                    bias_initializer=initializers.Constant(value=b[1]))(inputs_1)
hidden1_2 = Dense(1, name='Layer_2', activation='sigmoid',
                    kernel_initializer=initializers.Constant(value=a[2]),
                    bias_initializer=initializers.Constant(value=b[2]))(inputs_2)
hidden1_3 = Dense(1, name='Layer_3', activation='sigmoid',
                    kernel_initializer=initializers.Constant(value=a[2]),
                    bias_initializer=initializers.Constant(value=b[2]))(inputs_3)

# 包除積分(掛け算)
multiply1 = Multiply()([hidden1_1,hidden1_2])
multiply2 = Multiply()([hidden1_1,hidden1_3])
multiply3 = Multiply()([hidden1_2,hidden1_3])
multiply4 = Multiply()([hidden1_1,hidden1_2,hidden1_3])

# 結合
hidden2_m = concatenate([hidden1_1, hidden1_2, hidden1_3, multiply1, multiply2, multiply3, multiply4])

# 出力
predictions = Dense(1,name='Output',kernel_initializer=initializers.Ones(),
                         activation='relu')(hidden2_m)

# kernel_regularizer=keras.regularizers.l2(0.1), kernel_initializer=initializers.Zeros(),initializers.TruncatedNormal(mean=1.0, stddev=0.05, seed=None),
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

for i in range(12):
    d = model.layers[i]
    print(d.get_weights())