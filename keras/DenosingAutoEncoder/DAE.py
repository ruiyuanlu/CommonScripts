# coding=utf8

import numpy as np
# 指定随机数种子
np.random.seed(2053)
import matplotlib.pyplot as plt

import gzip

path = 'emnist-letters-train-images-idx3-ubyte.gz'
# 解压所需的库
import gzip

def unpack_gz(*file_lis):
    '''解压缩gz文件'''
    for f in file_lis:
        with open(f, 'rb') as fin:
            with open(f.replace('.gz', ''), 'wb') as fout:
                fout.write(gzip.decompress(fin.read()))

# 解压缩
unpack_gz(path)
# 加载数据集
import numpy as np
from EMNIST_Loader import EMNISTLoader
loader = EMNISTLoader(path.replace('.gz', ''), 'u1', (28, 28), 16)
y_train = loader.load()

# 样本归一化
y_train = y_train.astype('float32') / 255

# 随机采样噪声
noise = np.random.normal(loc=0, scale=1, size=y_train.shape)

# 加入噪声生成训练数据
x_train = 0.6 * np.clip(y_train + noise, 0, 1)

# 随机抽取4张图片用于对比
idx_lis = list(range(len(x_train)))
np.random.shuffle(idx_lis)
idx_lis = idx_lis[:4]

for i in range(8):
    # 绘制原始手写数字图像以及加入噪声的图像
    data = y_train if i < 4 else x_train
    plt.subplot(2, 4, i + 1)
    plt.imshow(data[idx_lis[i % 4]])
    plt.axis('off')
    plt.gray()

# plt.show()

# 构建降噪自动编码器: Denoising AutoEncoder
def get_DAE_model(input_shpae):
    from keras.models import Sequential
    from keras import layers
    dae = Sequential()
    # Encoder 部分
    # (28, 28, 1) -> (28, 28, 64)
    dae.add(layers.Conv2D(64, 3, input_shape=(28, 28, 1), padding='same'))
    dae.add(layers.Activation('relu'))
    # (28, 28, 64) -> (14, 14, 32)
    dae.add(layers.MaxPool2D((2, 2), padding='same'))
    dae.add(layers.Conv2D(64, 3, padding='same'))
    dae.add(layers.Activation('relu'))
    # (14, 14, 32)-> (7, 7, 32)
    dae.add(layers.MaxPool2D((2, 2), padding='same'))

    # Decoder 部分
    dae.add(layers.Conv2D(32, 3, padding='same'))
    dae.add(layers.Activation('relu'))
    # (7, 7, 32) -> (14, 14, 32)
    dae.add(layers.UpSampling2D((2, 2)))
    dae.add(layers.Conv2D(32, 3, padding='same'))
    dae.add(layers.Activation('relu'))
    # (14, 14, 32) -> (28, 28, 32)
    dae.add(layers.UpSampling2D((2, 2)))
    # (28, 28, 32) -> (28, 28, 1)
    dae.add(layers.Conv2D(1, 3, padding='same'))
    # 通过 sigmoid 函数将图像压缩为0-1之间的灰度图像
    dae.add(layers.Activation('sigmoid'))

    return dae
# 检查模型输出尺度
dae = get_DAE_model((28, 28, 1))
dae.summary()

# 扩展通道维度, 用于输入卷积层
x_train = x_train.reshape(-1, 28, 28, 1)
y_train = y_train.reshape(-1, 28, 28, 1)

# 编译模型, 使用MSE作为损失衡量指标
from keras.optimizers import SGD
# dae.compile(SGD(lr=0.2), loss='binary_crossentropy')
dae.compile(SGD(lr=2.0), loss='mse')

from keras.callbacks import TensorBoard

callback_lis = [
    # 使用 TensorBoard 绘制指标变化情况
    TensorBoard('tf_logs'),
]

# 训练模型 1
dae.fit(x_train, y_train,
        epochs=50,
        batch_size=64,
        callbacks=callback_lis)

# 输出模型复原的图像
y_pred = dae.predict(x_train)

# 构建第二层降噪自编码器
dae2 = get_DAE_model((28, 28, 1))
dae2.compile(SGD(lr=0.1), loss='mse')

dae2.fit(y_pred, y_train,
         batch_size=64,
         epochs=100)

# 以第一层降噪自编码器的输出作为输入进行降噪
y_pred = dae2.predict(y_pred)

# 复原维度用于显示图像
x_train = x_train.reshape(-1, 28, 28)
y_train = y_train.reshape(-1, 28, 28)
y_pred = y_pred.reshape(-1, 28, 28)

for i in range(12):
    # 绘制原始手写数字图像以及加入噪声的图像
    if i < 4:
        data = y_train
    elif i < 8:
        data = x_train
    else:
        data = y_pred
    plt.subplot(3, 4, i + 1)
    plt.imshow(data[idx_lis[i % 4]])
    plt.axis('off')
    plt.gray()

plt.show()