# coding=utf8

from keras.layers import BatchNormalization, Conv2D, Dense, Deconv2D
from keras.layers import Dropout, Reshape, LeakyReLU, Flatten

from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import Adam

from keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt

def plot_fake_imgs(epoch, batch, fakes):
    plt.cla()
    plt.clf()
    fakes = (127.5 * fakes + 127.5).astype('uint8').reshape(-1, 28, 28)
    width = height = int(np.sqrt(batch_size))
    img_mat = np.zeros((height * 28, width * 28))
    for i in range(height):
        for j in range(width):
            img_mat[i*28: (i + 1)*28, j*28: (j + 1)*28] = fakes[i * width + j]
    plt.imshow(img_mat)
    plt.axis('off')
    plt.gray()
    ax = plt.gca()
    fig = plt.gcf()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(f'pics/{epoch+1}_{batch+1}.png', bbox_inches=extent)


def plot_losses(epoch, g_losses, d_losses):
    plt.cla()
    plt.clf()
    plt.plot(range(len(g_losses)), g_losses)
    plt.plot(range(len(d_losses)), d_losses)
    plt.legend(['Generative Loss', 'Discriminative Loss'])
    plt.savefig(f'losses/{epoch+1}.png', pad_inches=0)


def create_generator(latent_dim, output_channels):
    '''
    创建生成模型
    '''
    generator = Sequential(name='generator')
    # 映射并转换维度
    generator.add(Dense(4 * 4 * 512, input_shape=(latent_dim,)))
    generator.add(Reshape((4, 4, 512)))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())

    # 使用上采样进行空间维度扩展
    # 4x4 -> 7x7 反卷积的空间尺度大小 = (input - 1) * strides + kernels
    # 因此有 (4 - 1) * 1 + 4 = 7
    generator.add(Deconv2D(64, 4, padding='valid'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())

    # 7x7 -> 14x14
    generator.add(Deconv2D(128, 3, strides=2, padding='same'))
    generator.add(BatchNormalization())
    generator.add(LeakyReLU())
    # 14x14 -> 28x28
    generator.add(Deconv2D(output_channels, 3, strides=2, padding='same', activation='tanh'))
    generator.summary()
    return generator


def create_discriminator(img_shape):
    '''
    创建判别模型
    '''
    discriminator = Sequential(name='discriminator')
    # 1通道扩展为64通道, 下采样到14x14
    discriminator.add(Conv2D(64, 5, strides=2, input_shape=img_shape, padding='same'))
    discriminator.add(LeakyReLU())

    # 下采样 7x7
    discriminator.add(Conv2D(32, 5, strides=2, padding='same'))
    discriminator.add(LeakyReLU())

    # 分类器层
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.summary()
    return discriminator

def train(batch_size, epochs=100):
    # 使用Adam作为生成器G和判别器D的优化器
    D_optimizer = Adam(1e-4, beta_1=0.5)
    G_optimizer = Adam(1e-4, beta_1=0.5, decay=1e-6)

    # 构建生成器G和判别器D, 这里只需对 D 进行编译
    G = create_generator(latent_dim, 1)
    D = create_discriminator(input_shape)
    D.compile(loss='binary_crossentropy', optimizer=D_optimizer)

    # 基于生成器和判别器构建GAN
    # GAN中, 判别器D作为一个权重冻结的层使用
    D.trainable = False
    GAN = Sequential([G, D], name='GAN')
    # 这里的编译对GAN和生成器G都进行了编译
    GAN.compile(loss='binary_crossentropy', optimizer=G_optimizer)

    plot_model(G, G.name + '.png')
    plot_model(D, D.name + '.png')
    plot_model(GAN, GAN.name + '.png')
    exit(0)

    # 构建训练所需的标签
    # 训练判别器的标签一半为真一半为假
    # 训练生成器的标签全为真, 这里真=0 假=1
    combined_labels = np.zeros((batch_size * 2, 1))
    combined_labels[batch_size:] = 1
    all_real_labels = np.zeros((batch_size, 1))

    # 训练主流程
    total_batchs = len(x_train) // batch_size
    for e in range(epochs):
        # 记录本轮损失变换的列表
        d_losses, g_losses = [], []
        for batch in range(total_batchs):
            # 随机生成隐变量噪声作为生成器的输入
            latent_noise = np.random.uniform(-1, 1, size=(batch_size, latent_dim))
            # 混合真实数据集和生成的假数据集
            real_imgs = x_train[batch * batch_size: (batch + 1) * batch_size]
            fake_imgs = G.predict(latent_noise)
            x = np.concatenate((real_imgs, fake_imgs))
            # 训练判别器
            d_loss = D.train_on_batch(x, combined_labels)
            # 采样新的噪声用于生成新样本
            latent_noise = np.random.uniform(-1, 1, (batch_size, 100))
            # 训练生成器
            g_loss = GAN.train_on_batch(latent_noise, all_real_labels)
            # 打印训练进度
            print(f'Epoch {e+1} batch {batch+1}/{total_batchs}',
                  'Generative Loss: %f' % g_loss,
                  'Discriminative Loss: %f' % d_loss, end='\r')
            # 记录损失变化情况
            d_losses.append(d_loss)
            g_losses.append(g_loss)

            # 绘制生成器生成的图像
            if (batch + 1) % 100 == 0:
                plot_fake_imgs(e, batch, fake_imgs)
        print()
        # 绘制本轮的损失变化
        plot_losses(e, g_losses, d_losses)


# 加载并归一化数据到(-1, 1)
(x_train, y_train), (_, _) = mnist.load_data()
x_train = (x_train - 127.5).reshape(-1, 28, 28, 1).astype('float32') / 127.5

# 隐变量维度
latent_dim = 100
input_shape = (28, 28, 1)

# 训练阶段参数
batch_size = 128
epochs = 100

# 创建文件夹
import os
if not os.path.exists('pics'):
    os.makedirs('pics')
if not os.path.exists('losses'):
    os.makedirs('losses')

# 开始训练
train(batch_size, epochs)