# coding=utf8

import os
from keras.datasets import cifar10
# 加载cifar10数据
(train_x, train_y), (test_x, test_y) = cifar10.load_data()
# 定义标签的描述信息
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

from matplotlib import pyplot as plt
# # 显示前10张图片并将其标签作为图像标题
# for i in range(12):
#     plt.subplot(4, 3, i+1)
#     # 关闭坐标轴
#     plt.axis('off')
#     x, y = train_x[i], train_y[i][0]
#     # 获取标签对应的描述
#     y_des = labels[y]
#     # 将图像数据输入Matplotlib中
#     plt.imshow(x)
#     # 将图像的标签和对应的描述作为标签
#     plt.title('%d: <%d-%s>' % (i+1, y, y_des))

# # 显示前12张图片
# plt.show()

# 数据预处理
import numpy as np
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

for data in [train_x, test_x]:
    # 逐通道去均值
    data -= np.mean(data, axis=(0, 1, 2))
    # 逐通道归一化
    data /= np.std(data, axis=(0, 1, 2))
    # 检查归一化结果, 均值接近于0, 方差接近于1
    print('输入数据均值:', data.mean())
    print('输入数据方差:', data.var())

# 将标签转换为onehot
from keras.utils import np_utils
train_y = np_utils.to_categorical(train_y, num_classes=10)
test_y = np_utils.to_categorical(test_y, num_classes=10)


from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPool2D, GlobalAvgPool2D
from keras.models import Sequential
from keras.regularizers import l2

rand_seed = 1837936012
np.random.seed(rand_seed)
使用序贯模型作为模型的容器
model = Sequential()
# ---------------------block_1-------------------------
# 卷积层
model.add(Conv2D(name='block_1_conv_1',
                 input_shape=(32, 32, 3),
                 filters=32, kernel_size=3, strides=1,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))
model.add(Conv2D(name='block_1_conv_2',
                 filters=32, kernel_size=3, strides=1,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))

# 池化层
model.add(MaxPool2D(name='block_1_maxpool', pool_size=2, strides=2, padding='same'))

# ---------------------block_2-------------------------
# 卷积层
model.add(Conv2D(name='block_2_conv_1',
                 filters=64, kernel_size=5, strides=2,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))
model.add(Conv2D(name='block_2_conv_2',
                 filters=64, kernel_size=5, strides=1,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))

# 池化层
model.add(MaxPool2D(name='block_2_maxpool', pool_size=2, strides=2, padding='same'))

# ---------------------block_3-------------------------
# 卷积层
model.add(Conv2D(name='block_3_conv_1',
                 filters=128, kernel_size=3, strides=1,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))
model.add(Conv2D(name='block_3_conv_2',
                 filters=128, kernel_size=3, strides=1,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))

# 池化层
model.add(MaxPool2D(name='block_3_maxpool', pool_size=2, strides=2, padding='same'))

# ---------------------block_4-------------------------
# 卷积层
model.add(Conv2D(name='block_4_conv_1',
                 filters=256, kernel_size=5, strides=1,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))
model.add(Conv2D(name='block_4_conv_2',
                 filters=256, kernel_size=5, strides=1,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))

# 池化层
model.add(MaxPool2D(name='block_4_maxpool', pool_size=2, strides=2, padding='same'))

# ---------------------block_5-------------------------
# 卷积层
model.add(Conv2D(name='block_5_conv_1',
                 filters=512, kernel_size=3, strides=2,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))
model.add(Conv2D(name='block_5_conv_2',
                 filters=512, kernel_size=3, strides=2,
                 activation='relu', padding='same',
                 kernel_regularizer=l2(2e-3),
                 kernel_initializer='he_normal'))

# 池化层
model.add(MaxPool2D(name='block_5_maxpool', pool_size=2, strides=2, padding='same'))
# ---------------------过渡层-------------------------
model.add(Flatten())
# ---------------------分类层-------------------------
model.add(Dense(4096, name='dense_2', activation='relu', kernel_initializer='he_normal'))
# 随机屏蔽神经元对抗过拟合
model.add(Dropout(rate=0.25))
model.add(Dense(10, name='classifier', activation='softmax', kernel_initializer='he_normal'))

# 检查模型结构
model.summary()

# 编译模型
from keras.optimizers import SGD
sgd = SGD(lr=1e-2, momentum=0.99, decay=1e-4, nesterov=True)
model.compile(sgd, 'categorical_crossentropy', metrics=['acc'])
# 训练模型30轮
history = model.fit(x=train_x, y=train_y, epochs=30,
                    batch_size=256,
                    validation_data=(test_x, test_y))

# 可视化训练结果
epochs = 30
train_acc, val_acc = history.history['acc'], history.history['val_acc']
train_loss, val_loss = history.history['loss'], history.history['val_loss']

plt.plot(range(1, epochs+1), train_acc, 'r-', range(1, epochs+1), val_acc, 'bo')
plt.legend(['train accuracy', 'val accuracy'])
plt.show()

plt.plot(range(1, epochs+1), train_loss, 'r-', range(1, epochs+1), val_loss, 'bo')
plt.legend(['train loss', 'val loss'])
plt.show()

input('>>> press any key to save the model...')
model.save('CIFAR10.h5')
exit(0)
引入Keras图像生成器
from keras.preprocessing.image import ImageDataGenerator
产生一个用于构造生成器的对象
为图像增加左右平移, 和翻转
generator = ImageDataGenerator(width_shift_range=0.125, height_shift_range=0.125,
                               vertical_flip=True)

构造训练数据生成器
生成训练集生成器
train_gen = generator.flow(train_x, train_y, batch_size=256)

history = model.fit_generator(train_gen, 
                    steps_per_epoch=150000 // 256 + 1,
                    epochs=30,
                    validation_data=(test_x, test_y))

# 可视化训练结果
epochs = 30
train_acc, val_acc = history.history['acc'], history.history['val_acc']
train_loss, val_loss = history.history['loss'], history.history['val_loss']

plt.plot(range(1, epochs+1), train_acc, 'r-', range(1, epochs+1), val_acc, 'bo')
plt.legend(['train accuracy', 'val accuracy'])
plt.show()

plt.plot(range(1, epochs+1), train_loss, 'r-', range(1, epochs+1), val_loss, 'bo')
plt.legend(['train loss', 'val loss'])
plt.show()

model.save('CIFAR10-argumt.h5')

from keras.models import load_model
model = load_model('CIFAR10.h5')
