# 说明

词语级别（Word-Level）的基于Keras的端到端英语到中文的机器翻译

字符级别的Keras端到端翻译可参考Keras的博客：

https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

以及对应的Github地址：https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

中英语料下载地址：http://www.manythings.org/anki/

# 执行流程说明

1. 首先执行params.py可以初始化所有文件夹

2. 然后执行preprocess.py用于加载语料库cmn.txt并进行中文英文预处理（生成索引序列和字典）

3. 最后执行MNT.py训练模型并输出样例句子中的结果

params.py: 定义相关参数

preprocess.py 定义和执行预处理方法和预处理流程

Models.py 定义编码器解码器模型结构

postprocess.py 用于MNT.py的译码输出截断

MNT.py 翻译流程的主流程代码
