# coding=utf8

from keras import layers
from keras import metrics
import keras.backend as K
from keras.models import Model, Input
from keras.datasets import mnist
from keras.utils import to_categorical

import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 图像的输入维度
input_shape = (28, 28, 1)

# 图像归一化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 图像维度扩展
x_train = x_train.reshape((-1,) + input_shape)
x_test = x_test.reshape((-1,) + input_shape)

# onehot 标签
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# 隐空间维度
latent_dim = 2

# 构建encoder
input_img = Input(shape=(28, 28, 1), name='Image_Input')
# CVAE
input_label = Input(shape=(10,), name='Input_Label')

x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)


shape_before_flattening = K.int_shape(x)[1:]

# 将多通道张量压缩为向量
x = layers.Flatten()(x)
# CVAE
x = layers.concatenate([x, input_label])
# VAE 编码器
x = layers.Dense(32, activation='relu')(x)

# 使用全连接层计算均值和log值
means = layers.Dense(latent_dim)(x)
z_logs= layers.Dense(latent_dim)(x)

# 构建高斯分布采样函数
def gaussian_sample(args):
    '''
    mean + exp(logs) * eps
    其中eps是随机生成的均值为0方差为1的随机张量
    '''
    means, logs = args
    eps = K.random_normal(shape=(K.shape(means)),
                          mean=0, stddev=1)
    return means + K.exp(logs / 2) * eps

# 采样并输出
z = layers.Lambda(gaussian_sample)([means, z_logs])

# CVAE z_laebl
z_label = layers.concatenate([z, input_label])

# CVAE 解码网络
decoder_input = Input(K.int_shape(z_label)[1:])
# 线性变换扩展维度
x = layers.Dense(np.prod(shape_before_flattening), activation='relu')(decoder_input)
# 变化维度, 以便可以输入到反卷积中
x = layers.Reshape(shape_before_flattening)(x)
# 反卷积
x = layers.Deconv2D(32, 3, strides=2, padding='same', activation='relu')(x)
# 生成灰度图
x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

# 解码器模型
decoder = Model(decoder_input, x)

decoder.summary()

# CVAE恢复图像
z_decoded = decoder(z_label)


def KL_loss(y_true, y_pred):
    '''
    KL 散度损失
    '''
    return -0.5 * K.mean(1 + z_logs - K.square(means) - K.exp(z_logs), axis=-1)

def reconstruction_loss(y_true, y_pred):
    '''
    图像重建损失 这里使用了2类交叉熵, 也可以使用MSE
    '''
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return metrics.binary_crossentropy(y_true, y_pred)

def VAE_loss(y_true, y_pred, alpha=1e-6):
    '''
    VAE 损失 KL散度损失需要乘以较小的系数 alpha，避免参数限制过度
    '''
    recon_loss = reconstruction_loss(y_true, y_pred)
    kl_loss = KL_loss(y_true, y_pred) * alpha
    return K.mean(recon_loss + kl_loss)

# CVAE
cvae = Model([input_img, input_label], z_decoded)

batch_size = 128

cvae.compile('rmsprop', loss=VAE_loss,
             metrics=[KL_loss, reconstruction_loss])

# CVAE 模型训练
cvae.fit([x_train, y_train_onehot], x_train,
         batch_size=batch_size,
         epochs=10,
         validation_data=([x_test, y_test_onehot], x_test))

# 构建条件采样编码
def conditional_vec(y, z=0):
    '''
    CVAE的解码器输入层包含了 (z, y)
    因此构造的vec维度为 latent_dim + 10
    vec 的大小需要与 InceptionV3_based_CVAE_decoder 方法中的 yz 变量对应
    '''
    vec = np.zeros((1, latent_dim + 10))
    # 隐变量, 未给定时仍为0不变
    vec[0, :latent_dim] = z
    # 标签y的onehot
    vec[0, latent_dim:][y] = 1
    return vec

# 采样绘图
from matplotlib import pyplot as plt
from scipy.stats import norm
# 采样 cnt * cnt 个数字
cnt = 20
# 图像变长像素数目
size = 28
# 矩阵填充以绘制无边框图像
mat = np.zeros((cnt * size, cnt * size))
# 高斯采样, 因为CVAE假定学习的是隐变量的高斯空间分布
x_grid = norm.ppf(np.linspace(0.05, 0.95, cnt))
y_grid = norm.ppf(np.linspace(0.05, 0.95, cnt))

# 绘制采样图像
for i, z1 in enumerate(x_grid):
    for j, z2 in enumerate(y_grid):
        z_sampled = np.array([z1, z2]).reshape(1, 2)
        # 生成 Conditional Vector
        z_vec = conditional_vec(2, z_sampled)
        z_decoded = decoder.predict(z_vec)
        # 用采样后的数字图像填充矩阵
        mat[i * size: (i + 1) * size,
            j * size: (j + 1) * size] = z_decoded.reshape(size, size)

# 生成最终图像
plt.figure()
plt.imshow(mat)
plt.gray()
plt.axis('off')


# 绘制样本编码分布, 首先构建编码器, 采样均值作为绘图的坐标
plt.figure()
# CVAE中增加标签作为输入
encoder = Model([input_img, input_label], means)
# 输出编码张量
means_encoded = encoder.predict([x_train, y_train_onehot]).reshape(-1, latent_dim)

# 绘制样本分布, 类别由y_train指定
# 并为每种类别指定不同的颜色
label_set = set(y_train)
from matplotlib import cm
colors = iter(cm.rainbow(np.linspace(0, 1, len(label_set))))

# 绘制类别为y的样本, 颜色为color
for y, color in zip(label_set, colors):
    idx = (y_train == y)
    plt.scatter(means_encoded[idx, 0], means_encoded[idx, 1],
                c=color, label=y, alpha=0.5, cmap='jet')
# 显示分布情况
plt.legend()
plt.show()