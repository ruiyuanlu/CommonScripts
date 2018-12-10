# coding=utf8
import os
import shutil
import re

# 路径设置
# ------------------------------------------------------
root_dir = os.path.dirname(os.path.abspath(__file__))
pre_dir = os.path.join(root_dir, 'pre_dir')
pic_dir = os.path.join(root_dir, 'pic_dir')
log_dir = os.path.join(root_dir, 'tf_log')
model_dir = os.path.join(root_dir, 'model_dir')
# Joint 模型存储路径
joint_weights_path = os.path.join(model_dir, 'Joint.h5')
joint_topology_path = os.path.join(model_dir, 'Joint_Topo.json')

# 平行语料文件路径
corpus_path = os.path.join(root_dir, 'cmn.txt')

# 语料设置
# ------------------------------------------------------
# 序列起始符 Start Of Sequence
BOS = '<BOS>'
# 序列终止符 End Of Sequence
EOS = '<EOS>'
# 源语言类型
src_lang = 'En'
# 目标语言类型
trgt_lang = 'Ch'
# 句子最大词语数目
maxlen = 20
# 目标语料句子最大长度
trgt_maxlen = 20
# 最大语料样本本数目
max_sample_num = 10

# 训练设置
# ------------------------------------------------------
# 使用反向源语言句子作为输入
src_reverse = True
# 随机数种子
seed = 123456
# 循环层嵌套深度
depth = 3
# 循环神经网络类型
rnn_type = 'gru' # lstm, gru, rnn
# 训练批大小设置
batch_size = 1
# 训练轮数
epoch = 300
# 验证集比例
val_split = 0.0


# 路径初始化方法
# ----------------------------------------------------------
def initialize_dirs(params_dict, overwrite=False):
    '''
    初始化所有文件夹
    根据 params.py 中设置的全局参数
    创建或覆盖文件夹
    '''
    assert isinstance(params_dict, dict), 'params_dict must be dict'
    params = params_dict.copy()
    # 删除内置的属性
    rm_keys = list(k for k in params.keys() if re.match('__(.*)__', k))
    for k in rm_keys:
        params.pop(k, None)
    # 遍历所有文件夹路径, 剔除变量名后缀中没有 _path _dir 的路径变量
    valid_dirs = (p for k, p in params.items() if (isinstance(p, (str, bytes, os.PathLike)) and
                                            k.endswith('_dir') and not os.path.ismount(p) and
                                            not os.path.isfile(p) and not os.path.islink(p)))

    # 初始化路径时排除当前文件夹
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    valid_dirs = list(os.path.abspath(p) for p in valid_dirs if p not in cur_dir)
    # 较短的路径优先创建, 长度相等时字典序小的优先创建
    valid_dirs.sort(key=lambda x: (len(x), x))
    for p in valid_dirs:
        if overwrite:
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)


if __name__ == '__main__':
    # 初始化所有文件夹
    initialize_dirs(locals())