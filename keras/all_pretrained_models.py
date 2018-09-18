# coding=utf8
# author: luruiyuan
# v0.1 date: 2018-09-18

# 常用的预训练的模型, 首次加载时, 需要下载预训练的权重, input_shape 是其默认形状, 可以修改
# Keras 根据是否加载顶层文件将权重文件分为 *.h5（包含顶层权重） 以及 *_notop.h5（无顶层权重）
# 因此首次执行时下载两种h5文件是必要的

from keras.applications import VGG16, VGG19
from keras.applications import Xception, InceptionV3
from keras.applications import InceptionResNetV2, ResNet50

VGG16_base = VGG16(include_top=True, input_shape=(224, 224, 3))
VGG19_base = VGG19(include_top=True, input_shape=(224, 224, 3))
Res50_base = ResNet50(include_top=True, input_shape=(224, 224, 3))
Xception_base = Xception(include_top=True, input_shape=(299, 299, 3))
InceptionV3_base = InceptionV3(include_top=True, input_shape=(299, 299, 3))
InceptionResNetV2_base = InceptionResNetV2(include_top=True, input_shape=(299, 299, 3))


# -------------------------------------------------------------------------
# 无顶层权重的网络
VGG16_top = VGG16(include_top=False, input_shape=(224, 224, 3))
VGG19_top = VGG19(include_top=False, input_shape=(224, 224, 3))
Res50_top = ResNet50(include_top=False, input_shape=(224, 224, 3))
Xception_top = Xception(include_top=False, input_shape=(299, 299, 3))
InceptionV3_top = InceptionV3(include_top=False, input_shape=(299, 299, 3))
InceptionResNetV2_top = InceptionResNetV2(include_top=False, input_shape=(299, 299, 3))


# 不太常用的预训练的模型, Keras 也提供预训练的权重的模型

from keras.applications import MobileNet
from keras.applications import DenseNet121, DenseNet169, DenseNet201
from keras.applications import NASNetLarge, NASNetMobile

Mobile_base = MobileNet(include_top=True, input_shape=(224, 224, 3))

Dense121_base = DenseNet121(include_top=True, input_shape=(224, 224, 3))
Dense169_base = DenseNet169(include_top=True, input_shape=(224, 224, 3))
Dense201_base = DenseNet201(include_top=True, input_shape=(224, 224, 3))

NASNetLarge_base = NASNetLarge(include_top=True, input_shape=(331, 331, 3))
NASNetMobile_base = NASNetMobile(include_top=True, input_shape=(224, 224, 3))

# -------------------------------------------------------------------------
# 无顶层权重的网络
Mobile_top = MobileNet(include_top=False, input_shape=(224, 224, 3))

Dense121_top = DenseNet121(include_top=False, input_shape=(224, 224, 3))
Dense169_top = DenseNet169(include_top=False, input_shape=(224, 224, 3))
Dense201_top = DenseNet201(include_top=False, input_shape=(224, 224, 3))
