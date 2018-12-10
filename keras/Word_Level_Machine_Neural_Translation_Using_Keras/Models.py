# coding=utf8

from keras.models import Model, Input
from keras import layers
from keras import backend as K


class MNTModel(object):
    '''
    Machine Neural Translation Model.
    机器神经翻译主体类
    '''
    
    # 网络名称到对应模型的映射
    name2layer = {
        'lstm': layers.LSTM,
        'gru': layers.GRU,
        'rnn': layers.SimpleRNN,
    }

    def __init__(self):
        super(MNTModel, self).__init__()


    def create_encoder_joint_model(self, src_max_len,
                                   src_vocab_size,
                                   trgt_vocab_size,
                                   units=32,
                                   rnn_type='lstm',
                                   encoder_emb_dim=100,
                                   encoder_depth=1,
                                   decoder_depth=1,
                                   mask_zero=True):
        '''
        创建 Ecoder 和 Ecoder-Decoder 联合训练模型.
        Decoder 训练阶段使用 teacher forcing: 训练数据比标签数据早一个timestep
        Args:
            src_max_len:        int 源语言句子中词语最大个数 (char-level时为字的最大个数)
            src_vocab_size:     int 源语言词汇表大小
            trgt_vocab_size:    int 目标语言词汇表大小
            units:              int 隐层神经元数目
            rnn_type:           int 编码器解码器RNN类型必须保持一致 lstm, gru 等, 由 name2layer 决定
            encoder_emb_dim:    int 编码器Embedding层输出维度
            encoder_depth:      int 编码器RNN层嵌套深度, 默认为1
            decoder_depth:      int 解码器RNN层嵌套深度, 默认为1
            mask_zero:          bool 是否使用0作为padding的掩码
                                    为True时Embedding层的0向量无效, 且 vocab_size += 1
        '''
        # 0 掩码表示 padding, 此时 Embedding 层的 0 向量无效, 因此 Embedding 索引从 1 开始
        if mask_zero:
            src_vocab_size += 1
            trgt_vocab_size += 1
        # 创建 Encoder 模型
        self.encoder = self._create_encoder(src_max_len, src_vocab_size, units, rnn_type,
                                encoder_emb_dim, encoder_depth, mask_zero)
        # 创建联合训练模型
        self.joint_model = self._create_joint(trgt_vocab_size, units, rnn_type, decoder_depth)
        return self.encoder, self.joint_model


    def _create_encoder(self, src_max_len,
                        src_vocab_size,
                        units,
                        rnn_type,
                        encoder_emb_dim,
                        encoder_depth,
                        mask_zero):
        '''
        创建 Encoder 的方法. 重复的参数说明见 create_encoder_joint_model

        Return:
            Model: 以词序列为输入, Encoder隐层特征为输出的Encoder模型
        '''
        # 创建 Encoder 模型: 序列输入层->Embedding->多层循环神经网络->输出隐层状态
        self.encoder_input = Input(shape=(src_max_len,), name='Source_Language_Input')
        encoder_embed = layers.Embedding(src_vocab_size, encoder_emb_dim, mask_zero=mask_zero)
        # Encoder 的最终输出不需要返回序列, 因此 return_sequences=False
        encoder_stack = self.create_recurrent_layers(units, rnn_type, encoder_depth, False)
        encoder_outputs = self.connect_layers(self.encoder_input, encoder_embed, *encoder_stack)
        # 注意: LSTM 有2个隐层状态 (h_t, c_t); GRU 只有一个隐层状态 h_t
        # 舍弃 Encoder 的输出, Encoder 的隐层状态即为 context vector. 参见: https://arxiv.org/pdf/1406.1078.pdf
        self.encoder_states = encoder_outputs[1:]
        return Model(self.encoder_input, self.encoder_states)


    def _create_joint(self, trgt_vocab_size,
                      units,
                      rnn_type,
                      decoder_depth):
        '''
        创建 Encoder-Decoder 联合训练模型的方法. 参数说明见 create_encoder_joint_model
        Decoder 采用 teacher forcing 进行训练, 因此输入数据比输出数据提前一个timestep
        Args:
            encoder_inputs: list 编码器模型输入层, 用于构建Decoder的训练模型
            encoder_states: list 编码器模型输出的隐层状态列表
                                GRU隐层状态有1个 h_t; LSTM 隐层状态有2个 h_t, c_t
                                编码器的隐层状态也被称为上下文向量 (context vector)
                                参见: https://arxiv.org/pdf/1406.1078.pdf
        '''
        # t-1时刻的词语 onehot 输入层, None 表示不定长的时间序列展开大小
        self.decoder_input = Input(shape=(None, trgt_vocab_size), name='Teacher_Word_Input')
        # Decoder 需要输出当前状态用于下一个词语的预测, 因此 return_sequences=True
        self.decoder_stack = self.create_recurrent_layers(units, rnn_type, decoder_depth, True)
        # Decoder 第一层循环网络需要以 Encoder 的状态初始化, 这样可以将Encoder编码的上下文的信息交给Decoder
        first_rnn, rest_layers = self.decoder_stack[0], self.decoder_stack[1:]
        # 注意这里的 initial_state 是 Encoder 直接交给 Decoder 的, 不是通过 Input 进行输入
        decoder_first_tensor = first_rnn(self.decoder_input, initial_state=self.encoder_states)
        # 连接其余循环层, 获得序列输出和隐层状态
        decoder_outputs = self.connect_layers(decoder_first_tensor, *rest_layers, initial_state=self.encoder_states)
        # decoder_states 只用于解码模型中, 因此这里舍弃
        decoder_output = decoder_outputs[0]
        # 使用Dense softmax作为最后一层输出词语概率, 这里可扩展为一系列线性层
        self.decoder_dense = layers.Dense(trgt_vocab_size, activation='softmax')
        decoder_output = self.decoder_dense(decoder_output)
        # 构建可训练的 Decoder 模型, 由 Encoder 和 Decoder 共同输入 t-1 时刻信息, 预测 t 时刻输出
        return Model([self.encoder_input] + [self.decoder_input],
                    decoder_output)


    def create_decoder(self):
        '''
        构建解码模型用于序列推断 (Inference)
        Args:
            encoder_states: list 编码器隐层状态列表
            joint: Model 解码器模型
        '''
        # Encoder context vector 输入层, 输入层用于 Inference Model
        states_inputs = [Input(batch_shape=K.int_shape(s), name=f'States_{i}') for i, s in enumerate(self.encoder_states)]
        # 接收 Teacher_Word_Input 输入的第一个循环层
        first_rnn = self.decoder_stack[0]
        # 将第一个循环层的初始状态改为从输入层获取, 其他不变
        first_tensor = first_rnn(self.decoder_input, initial_state=states_inputs)
        # 重新连接余下循环层的结构, 不连接全连接层
        decoder_outputs = self.connect_layers(first_tensor, *self.decoder_stack[1:], initial_state=states_inputs)
        # 在解码模型的输出中保留 decoder_states
        decoder_output, decoder_states = decoder_outputs[0], decoder_outputs[1:]
        # 连接全连接层输出概率, 这里可扩展为一系列线性层
        decoder_output = self.decoder_dense(decoder_output)

        # 构建解码模型, 该模型的输入为 Encoder 的内部状态以及 t-1 时刻的词语
        # 输出 t 时刻的词语. 其中 Encoder 内部状态是由输入层传入的, 而 Encoder-Decoder
        # 联合模型则是 Ecoder 直接传入; 解码模型的输出中包含 decoder_states
        #  而 Decoder 不包含, 这是 Inference 和 Decoder 的2个主要不同
        return Model([self.decoder_input] + states_inputs,
                    [decoder_output] + decoder_states)


    def create_recurrent_layers(self, units, layer_type='lstm', depth=1, return_sequences=False):
        '''
        创建多层循环神经网络

        Args:
            units: int 隐层神经元数量
            layer_type: str name2layer 中支持的网络名称
            depth: int 网络层的数量
            return_sequences: bool 是否需要多层循环层返回最终的序列

        Return:
            layer_lis: list 由 depth 个网络层组成的列表
        '''
        layer = self.name2layer.get(layer_type.lower(), layers.LSTM)
        depth = max(1, depth)
        # 中间层设置 return_sequences=True 用于下一循环神经网络
        layer_lis = [layer(units, return_sequences=True) for d in range(depth - 1)]
        # 最后一层根据本方法的参数确定是否返回序列, Encoder 为 False, Decoder 为 True
        layer_lis.append(layer(units, return_sequences=return_sequences, return_state=True))
        return layer_lis


    def connect_layers(self, first_layer, *layer_stack, initial_state=None):
        '''
        将传入的多层网络进行连接

        Args:
            first_layer: Layer Keras 模型输入层或中间层
            layer_stack: list 多层网络组成的参数列表
        
        Return:
            x: tensor, tuple 最后一层网络的输出
        '''
        x = first_layer
        for layer in layer_stack:
            x = layer(x, initial_state=initial_state) if initial_state else layer(x)
        return x