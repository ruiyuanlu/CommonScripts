# coding=utf8
import numpy as np

def save_model_topology_weights(model, topology_path, weights_path, encoding='utf8'):
    '''
    存储模型结构和权重
    '''
    with open(topology_path, 'w', encoding=encoding) as f:
        f.write(model.to_json())
    model.save(weights_path)


def decode_sequences(encoder, decoder, trgt_word2id, trgt_id2wrod,
                     *input_sequences, zero_mask=True):
    '''
    基于贪心算法, 根据输入序列解码给出输出序列
    '''
    from params import BOS, EOS, trgt_maxlen
    trgt_vocab_size = len(trgt_id2wrod) + 1 if zero_mask else len(trgt_id2wrod)

    for input_seq in input_sequences:
        # 对输入进行编码, 词语级别的解码需要reshape成数据集训练时的二维格式
        states = encoder.predict(np.reshape(input_seq, (1, -1)))
        # 由于 GRU 的状态只有一个, 因此返回的不是列表而是 numpy array
        # 此时需要放在列表中, 才能与LSTM的 states(2个) 兼容
        if isinstance(states, np.ndarray):
            states = [states]
        # 为Teacher-Forcing输入起始的Teacher字符
        trgt_seq_teacher = np.zeros((1, 1, trgt_vocab_size), dtype='float32')
        trgt_seq_teacher[0, 0, trgt_word2id[BOS]] = 1
        # 通过贪心算法逐次生成序列
        decoded_seq = []
        for _ in range(400):
            outputs = decoder.predict([trgt_seq_teacher] + states)
            # 获得输出的序列, 同时更新循环网络编码状态, 用于下一词语
            output, states = outputs[0], outputs[1:]
            # 根据输出进行采样, 并加入到解码结果decoded_seq中
            sampled_idx = np.argmax(output[0, -1, :])
            if zero_mask and sampled_idx == 0:
                # 跳过padding字符
                continue
            sampled_word = trgt_id2wrod[sampled_idx]
            decoded_seq.append(sampled_word)
            # 采样终止条件
            if (sampled_word == EOS or
                len(decoded_seq) > trgt_maxlen):
                break
            # 更新Teacher字符
            trgt_seq_teacher[0, 0, sampled_idx] = 1
        # 一次序列采样完成后, 返回已生成的序列
        yield decoded_seq


def reweight():
    pass