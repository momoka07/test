from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from keras.models import Model
from keras import Input
from keras.layers import Activation, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# 学習データとテストデータを取得する。
(_x_train_val, _y_train_val), (_x_test, _y_test) = mnist.load_data()
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

    x = Conv2D(32, (3,3), padding='same', name='conv1')(input)
    x = Activation(activation, name='act1')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)

    x = Conv2D(64, (3,3), padding='same', name='conv2')(x)
    x = Activation(activation, name='act2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)

    x = Conv2D(1, (3,3), padding='same', name='conv3')(x)
    x = Activation(activation, name='act3')(x)
    x = MaxPooling2D((3,3), name='pool3')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4, name='dense4')(x)
    x = Dense(64, name='dense5')(x)
    x = Activation(activation, name='act4')(x)
    x = Dense(10, name='dense6')(x)
    output = Activation('softmax', name='last_act')(x)

    model = Model(input, output)

    return model

model = model_functional_api()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])

# モデルの可視化
#keras.utils.plot_model(model, "my_CNN_model.png")

# 中間層の出力
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=[model.get_layer('pool3').output,model.get_layer('flatten').output])
intermediate_output1,intermediate_output2 = intermediate_layer_model.predict(x_train)

# モデルの可視化
#keras.utils.plot_model(intermediate_layer_model, "intermediate_model.png")

# 出力の確認
print(intermediate_output1)
print(intermediate_output2)

history=model.fit(x_train, y_train, validation_split=0.25, epochs=3, batch_size=128, verbose=1)