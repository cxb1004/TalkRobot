"""
生成词向量文件和Embadding.txt
1、读入csv文件
2、生成corpus.txt
3、生成word2Vec.bin
"""
"""
导入系统类库并设定运行根目录
"""
import os
import sys

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

"""
导入第三方类库
"""
import gensim
import csv
from gensim.models import word2vec

"""
导入自开发类库
"""
from common.log import Log

log = Log()

# 输入文件
file_labeled_train_csv = os.path.join(basePath, 'data\preProcess\labeledTrain.csv')
# 输出文件
file_corpus_txt = os.path.join(basePath, 'data\corpus.txt')
file_word2vec_bin = os.path.join(basePath, 'data\word2Vec.bin')

if os.path.isfile(file_corpus_txt):
    os.remove(file_corpus_txt)
    log.info('corpus.txt文件已经存在，自动删除')
if os.path.isfile(file_word2vec_bin):
    os.remove(file_word2vec_bin)
    log.info('ord2Vec.bin文件已经存在，自动删除')


log.info('根据labeledTrain.csv文件，生成词库文件corpus.txt...')
with open(file_corpus_txt, 'w', encoding='utf-8') as corpusFile:
    with open(file_labeled_train_csv, 'r', encoding='utf-8') as csvFile:
        allLines = csv.reader(csvFile)

        lineNum = 0
        for line in allLines:
            if not lineNum == 0:
                corpusFile.write(line[0])
            else:
                lineNum = 1

log.info('corpus.txt生成完毕...')

log.info('根据生成的corpus.txt文件，生成词向量文件...')
sentences = word2vec.LineSentence(file_corpus_txt)
# Word2Vec参数含义：
# - sentences:语料库，可以是列表，也可以直接从文件中读取
# - size: 词向量的维度，默认值是100,维度取值和语料库的大小有关
# - window：即词向量上下文最大距离，window越大，则和某一词较远的词也会产生上下文关系。默认值为5，一般的语料这个值推荐在[5；10]之间
# - sg：即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型；是1则是Skip-Gram模型；默认是0即CBOW模型
# - hs：即我们的word2vec两个解法的选择了。如果是0， 则是Negative Sampling；是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling
# - min_count：需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值
# - iter：随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
vec_size = 100
vec_window = 6
vec_min_count = 2
vec_sg = 1
vec_iter = 5

log.info('生成中...')
model = gensim.models.Word2Vec(sentences,
                               size=vec_size,
                               window=vec_window,
                               min_count=vec_min_count,
                               sg=vec_sg,
                               iter=vec_iter)
model.wv.save_word2vec_format(file_word2vec_bin, binary=True)
log.info('生成完毕')
