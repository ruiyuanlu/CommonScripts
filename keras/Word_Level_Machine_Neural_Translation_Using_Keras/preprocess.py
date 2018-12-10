# coding=utf8
import os
import json
import numpy as np

# 导入序列起始符和终止符
from params import BOS, EOS

def load_file(path, encoding='utf8'):
    '''
    加载平行语料文件, 每行一对平行语料, 中间制表符分隔
    http://www.manythings.org/anki/
    '''
    src_lis, trgt_lis = [], []
    with open(path, encoding=encoding) as f:
        for line in f:
            src, trgt = line.strip().split('\t')
            src_lis.append(src)
            trgt_lis.append(trgt)
    return src_lis, trgt_lis


def same_shuffle(*data_lis, seed=123456):
    '''
    对所有语料采用相同的顺序进行随机打乱
    '''
    idx = np.arange(min(len(d) for d in data_lis), dtype=int)
    np.random.seed(seed)
    np.random.shuffle(idx)
    return tuple(np.asarray(d)[idx] for d in data_lis)


def English_texts_numerical(str_txt_lis,
                            num_words=None,
                            reverse=False,
                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                            return_generator=True):
    '''
    英文预处理方法: 英文词语型符化->统计词频->转换为整数序列
    
    Args:
        str_txt_lis: list 英文字符串构成的列表, 每个字符串表示一篇文章
        num_words: int 出现频率最高的前 num_words 个词语数目
        reverse: bool 是否返回反向序列
        filters: str 过滤字符, 不能出现的过滤字符
        return_generator: bool Ture 返回序列生成器, False 返回列表
    
    Returns:
        word2id: dict 词语到索引的映射字典
        id2word: dict 索引到词语的映射字典
        word_cnts: dict 词语的出现次数字典
        seq: generator, list 将英文字符串转换为整数列表的生成器或列表
                        return_generator 决定其返回类型
    '''
    from keras.preprocessing.text import Tokenizer
    # 使用Keras自带的Tokenizer进行型符化
    tokenizer = Tokenizer(num_words, filters)
    tokenizer.fit_on_texts(str_txt_lis)
    # 按照词频从高到底构建词语与索引之间映射的字典
    word2id = convert_cnts_to_index(tokenizer.word_counts)
    id2word = {v: k for k, v in word2id.items()}
    # 由于修改了Keras的Tokenizer, 因此需要将修改后的字典赋值给Tokenizer
    tokenizer.word_index = word2id
    # 由于没有手动对英文序列进行分词, 因此仍然需要使用 Tokenizer 生成索引序列
    seq_token_gen = tokenizer.texts_to_sequences_generator(str_txt_lis)
    # 为Keras的Tokenizer添加序列的起始符和终止符
    rev = lambda seq, reverse: list(reversed(seq)) if reverse else seq
    seq_gen = ([word2id[BOS]] + rev(seq, reverse) + [word2id[EOS]] for seq in seq_token_gen)
    seq = seq_gen if return_generator else list(seq_gen)
    return word2id, id2word, tokenizer.word_counts, seq


def Chinese_texts_numerical(str_txt_lis,
                            num_words=None,
                            reverse=False,
                            filters='?？/。!！"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                            return_generator=True,
                            min_cnt=None):
    '''
    中文文本序列数值化理方法: 分词->统计词频->转换为整数序列
    
    Args:
        str_txt_lis: list 中文字符串列表, 每个字符串表示一篇文章
        num_words: int 出现频率最高的前 num_words 个词语数目
        reverse: int 是否返回反向序列
        return_generator: bool Ture 返回序列生成器, False 返回列表
        min_cnt: int 词频下限, 低于此频率的词语被视为稀有词, 默认为 1 (所有词频都保留)
    
    Returns:
        word2id: dict 词语到索引的映射字典
        id2word: dict 索引到词语的映射字典
        word_cnts: dict 词语的出现次数字典
        seq: generator, list 将中文字符串转换为整数列表的生成器或列表
                        return_generator 决定其返回类型
    '''
    import jieba
    from collections import OrderedDict
    # 构建过滤器
    filters = str.maketrans({f: None for f in filters})
    # 分词
    splited = [list(c.translate(filters) for c in jieba.cut(s)) for s in str_txt_lis]
    # 统计词频
    word_cnts = OrderedDict()
    for seq in splited:
        for w in seq:
            word_cnts[w] = word_cnts.get(w, 0) + 1
    # 按照词频从高到底构建词语与索引之间映射的字典
    word2id = convert_cnts_to_index(word_cnts)
    id2word = {v: k for k, v in word2id.items()}
    # 将字符序列转换为整数序列
    seq_gen = sequence_generator(splited, word2id, reverse, word_cnts, num_words, min_cnt)
    seq = seq_gen if return_generator else list(seq_gen)
    return word2id, id2word, word_cnts, seq


def convert_cnts_to_index(word_cnts):
    '''
    将词频统计字典转换为词语到索引的映射
    Keras Tokenizer 没有预置的终止符

    Args:
        word_cnts: dict 词频统计字典
    
    Return:
        word2id: 词语到索引的字典
    '''
    # 整数 0, 1, 2保留, 词汇索引从 3 开始.
    # 0 表示 padding, 1 表示序列起始符 BOS, 2 表示表示序列结束符 EOS
    wcounts = list(word_cnts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    word2id = {w: i + 3 for i, (w, _) in enumerate(wcounts)}
    word2id[BOS], word2id[EOS] = 1, 2
    return word2id


def Chinese_Traditional_Simple_converter(str_lis,
                                         mode='t2s',
                                         return_generator=True):
    '''
    中文繁简转换, 基于iNLP库
    
    Args:
        str_lis: list 中文字符串列表, 每个字符串表示一篇文章或一个词
        mode: str 转换模式 t2s (Traditional to Simple) = 繁转简
                   s2t (Simple to Traditional) = 简转繁
                   当遇到未知mode时, 默认使用繁转简
        return_generator: bool Ture 返回序列生成器, False 返回列表
    
    Return:
        seq: generator, list 将中文字符串转换为整数列表的生成器或列表
                            return_generator 决定其返回类型
    '''
    from inlp.convert import chinese
    mode2convert = {
        't2s': chinese.t2s,
        's2t': chinese.s2t,
    }
    converter = mode2convert.get(mode.lower(), chinese.t2s)
    seq_gen = (converter(s) for s in str_lis)
    seq = seq_gen if return_generator else list(seq_gen)
    return seq


def sequence_generator(seq_lis,
                       word_index,
                       reverse=False,
                       word_cnts=None,
                       num_words=None,
                       min_cnt=None):
    '''
    整数序列生成器. 索引超过 num_words 的词语
    以及词频小于 min_cnt 的词语会被视为稀有词并且忽略

    Args:
        seq_lis: Iterable 可迭代序列对象
        word_index: dict 词语到索引的字典
        reverse: bool 是否生成反序的序列, 默认 False
        word_cnts: dict 词语到词频的字典
        num_words: int 词语索引上限, 索引超过时视为稀有词
        min_cnt: int 词频下限, 词频低于时视为稀有词
    '''
    for seq in seq_lis:
        # 添加序列起始符
        vect = []
        for w in seq:
            i = word_index.get(w)
            if i is not None:
                if num_words and i > num_words:
                    continue
                if word_cnts is not None:
                    cnt = word_cnts.get(w, 0)
                    if min_cnt is not None and cnt < min_cnt:
                        continue
                vect.append(i)
        if reverse:
            vect = list(reversed(vect))
        # 为序列添加起始符和结束符
        yield [word_index[BOS]] + vect + [word_index[EOS]]


def save_preprocess(language, dirpath, word_cnts, sequences):
    '''
    存储预处理结果: 词频字典存储为json文件
                   词语索引序列存储为npz文件
    
    Args:
        language: str 预处理的语言名称或简写
                      作为前缀出现在存储的json和npz文件中
        dirpath:   str  预处理文件所在文件夹
        word_cnts: dict 词语出现次数的字典
        sequences: list, tuple, Iterable 
                   词语索引序列组成的列表
                   或可生成词语索引序列的迭代器
    '''
    # 词频字典的 json 文件路径
    json_path = os.path.join(dirpath, f'{language}_word_cnts.json')
    # 整数序列数组存储路径
    npy_path = os.path.join(dirpath, f'{language}_sequence')
    with open(json_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(word_cnts, ensure_ascii=False))
    sequences = sequences if isinstance(sequences, (list, tuple)) else list(sequences)
    np.save(npy_path, sequences)


def load_preprocess(language, dirpath, max_sample_num=None, 
                    num_words=None, return_generator=True):
    '''
    加载预处理结果
    
    Args:
        language: str 预处理的语言名称或简写
                      作为前缀出现在存储的json和npz文件中
        dirpath:  str  预处理文件所在文件夹
        max_sample_num: None, int 加载样本的最大数目, 防止内存不足.
                        None 表示全部加载, 传入大于0的整数N时加载前N个样本
        num_words: int 出现频率最高的前 num_words 个词语数目
        return_generator: bool Ture 返回序列生成器, False 返回列表
    
    Returns:
        word2id: dict 词语到索引的字典
        id2word: dict 索引到词语的字典
        word_cnts: OrderedDict 词频字典
        seq: generator, list 序列生成器或列表
                        return_generator 决定其返回类型
    '''
    # 词频字典的 json 文件路径
    json_path = os.path.join(dirpath, f'{language}_word_cnts.json')
    # 整数序列数组存储路径
    npy_path = os.path.join(dirpath, f'{language}_sequence.npy')
    # 加载词频字典
    with open(json_path, encoding='utf8') as f:
        word_cnts = json.loads(f.read().strip())
    # 整数 0, 1 保留, 词汇索引从 2 开始. 0 表示 padding, 1 表示序列结束符 EOS
    word2id = convert_cnts_to_index(word_cnts)
    id2word = {v: k for k, v in word2id.items()}
    # 加载索引序列
    sequences = np.load(npy_path)
    if max_sample_num is not None and max_sample_num > 0:
        sequences = sequences[:max_sample_num]
    seq = (s for s in sequences) if return_generator else sequences
    return word2id, id2word, word_cnts, seq


def encode_teacher_forcing(sequences, vocab_size, mask_zero=True):
    '''
    对目标语言序列按照索引进行teacher forcing 原则进行 onehot 编码.
    其中 teacher 比 stu 领先一个词 (1个timestep)

    Args:
        sequences: 二维list或ndarray 经过padding的词语索引序列
        vocab_size: int 词汇表大小
        mask_zero: bool 是否以0作为padding掩码, True时vocab_size += 1

    Return:
        teacher: ndarray (num_sample, maxlen, vocab_size) 的数组, 比stu领先1个timestep
                         teacher 用于训练 decoder 时的输入
        stu:     ndarray (num_sample, maxlen, vocab_size) 的数组, 比teacher落后1个timestep
                         stu 用于训练 decoder 时的输出
    '''
    assert np.ndim(sequences) == 2, f'sequences dim must be 2. {np.ndim(sequences)} Found.'
    from keras.utils import to_categorical
    if mask_zero:
        vocab_size += 1
    teacher = np.array(list(map(lambda s: to_categorical(s, vocab_size), sequences)), dtype=bool)
    stu = np.zeros(teacher.shape, dtype=bool)
    # teacher 比 stu 提前 1 个 timestep
    stu[:, :-1] = teacher[:, 1:]
    return teacher, stu


if __name__ == '__main__':
    from params import corpus_path
    # 加载源语言和目标语言
    src_lis, trgt_lis = load_file(corpus_path)
    # 随机洗牌
    from params import seed, src_reverse
    src_lis, trgt_lis = same_shuffle(src_lis, trgt_lis, seed=seed)
    # 英文序列处理, 将文本转换为数字序列
    En_word2id, En_id2word, En_word_cnts, En_seq = English_texts_numerical(src_lis, reverse=src_reverse, return_generator=False)
    # 中文繁简转换
    trgt_lis = Chinese_Traditional_Simple_converter(trgt_lis, return_generator=True)
    # 中文序列处理, 将文本转换为数字序列
    Ch_word2id, Ch_id2word, Ch_word_cnts, Ch_seq = Chinese_texts_numerical(trgt_lis, return_generator=False)
    # 输出预处理结果进行检查
    for i in range(10):
        en, ch = En_seq[i], Ch_seq[i]
        print(f'{en}\t{ch}')
        print(' '.join(map(En_id2word.get, en)), '\t', ' '.join(map(Ch_id2word.get, ch)))

    # 存储预处理结果
    from params import pre_dir
    from params import src_lang, trgt_lang
    
    save_preprocess(src_lang, pre_dir, En_word_cnts, En_seq)
    save_preprocess(trgt_lang, pre_dir, Ch_word_cnts, Ch_seq)

    # 加载预处理结果进行检查
    En_word2id, En_id2word, En_word_cnts, En_seq = load_preprocess(src_lang, pre_dir, return_generator=False)
    Ch_word2id, Ch_id2word, Ch_word_cnts, Ch_seq = load_preprocess(trgt_lang, pre_dir, return_generator=False)

    # 输出预处理结果进行检查
    print('-' * 60)
    for i in range(10):
        en, ch = En_seq[i], Ch_seq[i]
        print(f'{en}\t{ch}')
        print(' '.join(map(En_id2word.get, en)), '\t', ' '.join(map(Ch_id2word.get, ch)))