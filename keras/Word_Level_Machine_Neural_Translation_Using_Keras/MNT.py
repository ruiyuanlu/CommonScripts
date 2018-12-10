# coding=utf8
'''
MNT.py <-> Machine Neural Translation Using Keras
author: luruiyuan
'''

# 导入预处理相关参数和模型相关参数
from params import pre_dir
from params import src_lang, trgt_lang
from params import max_sample_num
from preprocess import load_preprocess

# 加载预处理结果
(En_word2id, En_id2word,
    En_word_cnts, En_seq) = load_preprocess(src_lang, pre_dir, max_sample_num,
                                            return_generator=False)
(Ch_word2id, Ch_id2word,
    Ch_word_cnts, Ch_seq) = load_preprocess(trgt_lang, pre_dir, 
                                            max_sample_num, return_generator=False)

# 序列参数 maxlen
from params import maxlen
from keras.preprocessing.sequence import pad_sequences
# 使用Keras自带的序列填充方法. padding和截断都针对序列尾部, 因此选择 post
X = pad_sequences(En_seq, maxlen=maxlen, padding='post', truncating='post')
Y = pad_sequences(Ch_seq, maxlen=maxlen, padding='post', truncating='post')

# 对预测序列按照 Teacher-Forcing 原则进行编码
from preprocess import encode_teacher_forcing
Y_teacher, Y_stu = encode_teacher_forcing(Y, len(Ch_word2id), True)


# 首先创建 Encoder Decoder 模型
from params import depth, rnn_type
from Models import MNTModel
mnt = MNTModel()
encoder, joint_model = mnt.create_encoder_joint_model(src_max_len=maxlen, src_vocab_size=len(En_word2id),
                                                      trgt_vocab_size=len(Ch_word2id), rnn_type=rnn_type,
                                                      encoder_depth=depth, decoder_depth=depth, mask_zero=True)

# 训练 Encoder-Decoder Joint 模型
from params import batch_size, epoch, val_split, log_dir
from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.optimizers import RMSprop

callback_lis = [
    TensorBoard(log_dir),
    LearningRateScheduler(lambda epoch, lr: 0.01 if epoch <= 100 else 0.001)
]

joint_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
joint_model.fit([X, Y_teacher], Y_stu,
                batch_size=batch_size,
                epochs=epoch,
                validation_split=val_split,
                callbacks=callback_lis)

# 存储联合模型
from params import joint_topology_path, joint_weights_path
from postprocess import save_model_topology_weights
save_model_topology_weights(joint_model, joint_topology_path, joint_weights_path)

# 训练完成后, 构建解码模型用于推断
from postprocess import decode_sequences
decoder = mnt.create_decoder()

test_seq = X[:100]
for seq, decode_seq in zip(test_seq, decode_sequences(encoder, decoder,
                                   Ch_word2id, Ch_id2word, *test_seq)):
    print('输入测试序列:', [En_id2word[idx] for idx in seq if idx > 0])
    print('输出解码结果:', ' '.join(decode_seq))


# 绘制模型结构图
from keras.utils import plot_model
from params import pic_dir
import os
plot_model(encoder, os.path.join(pic_dir, 'Encoder.png'))
plot_model(joint_model, os.path.join(pic_dir, 'Joint.png'))
plot_model(decoder, os.path.join(pic_dir, 'Decoder.png'))