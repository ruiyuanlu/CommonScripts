# coding=utf8
import os
import numpy as np

class EMNISTLoader(object):
    '''EMNIST数据加载器'''

    def __init__(self, path, dtype, sample_size, offset):
        '''EMNIST数据加载器初始化函数

        Args:
            data_path: str 文件路径
            dtype: str 数据类型描述字符串, 与numpy兼容
            sample_size: iterable 数据文件中每个样例的每个维度大小 以字节计算
            offset: int 文件头偏移量 以字节计算
        '''
        self.path = os.path.realpath(path)
        self.dtype = dtype
        self.sample_size = sample_size
        self.offset = offset
    
    def load(self):
        '''加载数据'''
        data = np.fromfile(self.path, dtype=self.dtype)[self.offset:]
        return data.reshape(-1, *self.sample_size)
