"""

"""
import os
import sys

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

import csv
from gensim.models import word2vec
import gensim
import pandas as pd
import numpy as np
from collections import Counter
import json
import tensorflow.compat.v1 as tf
import datetime

from common.log import Log
from common.utils import removeFileIfExists

log = Log()


# 全局变量
# 查询标签
# 标签和数字id的对应转换，另一块代码在_genVocabulary中，重构的时候选择一个方案，然后两个地方选择一处修改
def getTagDictory():
    tagFile = os.path.join(basePath, 'data/tag_info.csv')
    labelinfo = {}
    # idx = 0
    with open(tagFile, 'r', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if reader.line_num != 1:
                labelinfo.update({row[0]: row[0]})
                # idx = idx + 1
    log.info('读入标签{}个'.format(labelinfo.__len__()))
    return labelinfo


# 定义标签和对应的ID，用于打标签
LABEL_ID_DICT = getTagDictory()
# 反转标签的ID和标签值，用于查询
ID_LABEL_DICT = {v: k for k, v in LABEL_ID_DICT.items()}

# 读入数据：打过标签的数据文件 content / label_id / label
file_labeled_train_data_csv = os.path.join(basePath, 'data/labeledTrainData.csv')

# 输出数据：词向量文件
bin_word2Vec = os.path.join(basePath, 'data/word2Vec.bin')

# 词向量维度参数设置
VECTOR_SIZE = 200


def generateWord2Vector(_word2VecFile):
    """
    根据语料库文件生成词向量文件
    :param _word2VecFile词向量文件的生成路径
    :return:
    """
    log.info("开始生成词向量文件：{}".format(bin_word2Vec))
    # 生成同名的txt临时文件备查
    _temp_file = _word2VecFile.replace('.bin', '.tmp')
    # 确认临时文件是否存在
    if not os.path.isfile(_temp_file):
        log.error('文件不存在，请确认：{}'.format(_temp_file))
        raise FileNotFoundError('文件不存在，请确认：{}'.format(_temp_file))

    # word2Vec.bin文件如果已经存在，删除
    removeFileIfExists(_word2VecFile)

    '''
    gensim.models.Word2Vec参数含义：
        sentences=None: 供训练的句子，可以使用简单的列表，但是对于大语料库，建议直接从磁盘/网络流迭代传输句
        corpus_file=None: LineSentence格式的语料库文件路径， [["cat", "say", "meow"], ["dog", "say", "woof"]]
        size=100: 词向量的维度，默认100，取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度
        【可调】window=5: 词向量上下文最大距离,默认值为5,window越大，则和某一词较远的词也会产生上下文关系。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间
        min_count=5: 需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值
        sg=0: 默认是0。word2vec两个模型的选择，如果是0，则是CBOW模型，是1则是Skip-Gram模型
        cbow_mean=1:  0: 使用上下文单词向量的总和; 1: 使用均值，适用于使用CBOW
        hs=0: 默认是0。word2vec两个解法的选择，如果是0，则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。
        negative=5: 使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间
        ns_exponent=0.75:  负采样分布指数。1.0样本值与频率成正比，0.0样本所有单词均等，负值更多地采样低频词
        iter=5: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
        alpha=0.025: 在随机梯度下降法中迭代的初始步长。默认是0.025。
        min_alpha=0.0001: 随着训练的进行，学习率线性下降到min_alpha。 对于大语料，需要对alpha, min_alpha,iter一起调参
        hashfxn=<built-in function hash>:  哈希函数用于随机初始化权重，以提高训练的可重复性
        null_word=0: 
        trim_rule=None: 词汇修剪规则，指定某些词语是否应保留在词汇表中，修剪掉或使用默认值处理
        sorted_vocab=1: 如果为1，则在分配单词索引前按降序对词汇表进行排序
        workers=3:  训练模型时使用的线程数,worker参数只有在安装了Cython后才有效. 没有Cython的话, 只能使用单核
        batch_words=10000: 每一个batch传递给线程单词的数量
        max_final_vocab=None:自动选择匹配的min_count将词汇限制为目标词汇大小
        max_vocab_size=None:  词汇构建期间RAM的限制; 如果有更多的独特单词，则修剪不常见的单词。 每1000万个类型的字需要大约1GB的RAM
        seed=1:  随机数发生器种子
        sample=0.001: 高频词随机下采样的配置阈值，范围是(0,1e-5)
        compute_loss=False: 如果为True，则计算并存储可使用get_latest_training_loss()检索的损失值
        callbacks=():  在训练中特定阶段执行回调序列
    '''
    sentences = word2vec.LineSentence(_temp_file)
    # 【注意】这个size可以改动，但是需要和补齐的Embedding数据保持一致，否则Embedding会有{200,200,100,...}的维度不一致的情况，并在
    #  return vocab, np.array(wordEmbedding) 代码处抛出警告
    vec_size = VECTOR_SIZE
    vec_window = 6
    vec_min_count = 2
    vec_sg = 1
    vec_iter = 5
    model = gensim.models.Word2Vec(sentences,
                                   size=vec_size,
                                   window=vec_window,
                                   min_count=vec_min_count,
                                   sg=vec_sg,
                                   iter=vec_iter)
    model.wv.save_word2vec_format(_word2VecFile, binary=True)
    log.info("词向量文件生成完毕：{}".format(bin_word2Vec))


def generateWord2VectorFile(_sourceFile, _word2VecFile):
    """
    读入语料数据，清洗，并生成word2Vec.bin
    1、读入语料数据
    2、清洗数据（包括去除标点符号、去掉html标签、分词、空格分隔等）
    3、进行中文分词操作
    4、同目录下生成corpus.txt作为可视化的中间文件
    5、根据corpus.txt生成word2Vec.bin文件
    :return: 0：执行成功； 9：执行失败
    """
    # 检查数据源文本是否存在
    if not os.path.isfile(_sourceFile):
        log.error("文件不存在，请确认:{}".format(_sourceFile))
        raise FileNotFoundError('文件不存在，请确认：{}'.format(_sourceFile))

    # 生成同名的临时文件备查
    temp_file = _word2VecFile.replace('.bin', '.tmp')
    removeFileIfExists(temp_file)
    removeFileIfExists(_word2VecFile)

    log.info("开始生成临时语料库文件：{}".format(temp_file))
    with open(temp_file, 'w', encoding='utf-8', newline='') as tmpFile:
        with open(_sourceFile, 'r', encoding='utf-8') as csvFile:
            # 读取labeledTrainData.csv文件内容
            csvData = csv.reader(csvFile)
            for line in csvData:
                # 绕过第一行的标题
                if not csvData.line_num == 1:
                    content = line[0]
                    # 这里必须要加\n,否则生成的词向量内容就是一行文本的词向量，
                    tmpFile.write(content + '\n')
    log.info("临时语料库文件生成完毕：{}".format(temp_file))

    generateWord2Vector(_word2VecFile)


class TrainingConfig(object):
    """
    训练用的配置
    """
    # epoches: 使用训练集的全部数据对模型进行你个一次完整训练，被称之为‘一代训练’
    # 对全体数据进行4次训练
    epoches = 4
    # 每运行100条记录的时候，进行一次评估总结，输出评估结果到summary目录
    evaluateEvery = 100
    # 每运行100条，进行一次checkpoint
    checkpointEvery = 100
    # 学习速率参数 https://blog.csdn.net/sinat_29957455/article/details/78387716
    # 默认的参数就是0.001，梯度*学习率就是模型更新不步长，步长大了影响模型参数的最终精准度；步长小了影响模型训练的时间和效率。
    learningRate = 0.001


class ModelConfig(object):
    """
    模型的配置
    """
    embeddingSize = 200
    # ？？ LSTM结构的神经元个数
    hiddenSizes = [256, 128]
    # 过拟合参数:
    # 在不同的训练过程中，丢弃掉一部分的神经元，即让一部分神经元以一定的概率停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
    # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
    dropoutKeepProb = 0.5
    # ？？
    l2RegLambda = 0.0


class Config(object):
    """
    主要配置参数，包括运行
    """
    sequenceLength = 13  # 取了所有序列长度的均值
    batchSize = 128

    # 【待重构】
    # 原始数据源
    # dataSource = os.path.join(basePath, "data/labeledTrainData.csv")
    dataSource = file_labeled_train_data_csv

    # 【待重构】
    # 暂时先不考虑停用词
    # stopWordSource = os.path.join(basePath, "data/english")
    stopWordSource = None

    # 【待重构】
    # 这里我们用到的是多分类，不是二分类，参考csv文件中有10个分类，因此设置为10
    # numClasses = 11
    numClasses = len(LABEL_ID_DICT)

    # 训练集的比例，即80%用于训练，20%用于测试
    rate = 0.95

    # 初始化训练时候用的参数
    training = TrainingConfig()

    # 初始化模型参数
    model = ModelConfig()


class Dataset(object):
    """
    数据预处理的类，生成训练集和测试集
    """

    def __init__(self, config):
        """
        根据config来初始化Dataset所用到功能的参数
        :param config:
        """
        # Dataset类里面自己定义一个config对象，从外面传入config赋值
        self.config = config
        # 设置原始数据文件
        self._dataSource = config.dataSource
        # 设置停用词文件(暂时先不考虑停用词)
        self._stopWordSource = config.stopWordSource
        # 【待重构】
        # 设置序列长度200
        self._sequenceLength = config.sequenceLength
        # 【待重构】
        # 设置特征数量200
        self._embeddingSize = config.model.embeddingSize
        # 【待重构】
        # 设置批处理数量128（用于多次运行，修正模型参数）
        self._batchSize = config.batchSize
        # 设置训练集的比例，即80%的数据用来进行训练，20%数据用于测试
        self._rate = config.rate

        # 初始化停用词字典，在_readStopWord()进行初始化
        self._stopWordDict = {}

        # 训练集 review指的是文本内容，label是标签
        self.trainReviews = []
        self.trainLabels = []

        # 测试集 review指的是文本内容，label是标签
        self.evalReviews = []
        self.evalLabels = []

        # 训练集中的词向量
        self.wordEmbedding = None

        # 标签列表
        self.labelList = []

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """
        # 这里如果出现编码错误问题，一般来说是这个csv文件被wps工具打开并保存过，可以重新复制一下文件即可
        df = pd.read_csv(filePath, encoding='utf_8')

        # 这里是把标签从原始数据文件中提取出来
        # 数据文件中：
        # 第一列：文本内容
        # 第二列：标签ID
        # if self.config.numClasses == 1:
        #     labels = df["sentiment"].tolist()
        # elif self.config.numClasses > 1:
        # 【待优化】labels数据不应该从语料库里面拿，
        labels = df["label_id"].tolist()
        review = df["content"].tolist()
        # 【已变更】原有功能是每一行去除前后空格之后，按空格划分，即对英文进行分词操作
        # 而中文的分词需要特殊处理，所以这里要前置处理，在生成csv文件的时候，就把分词结果写入到content里面
        # 因此在生成corpus.txt文件的时候，同时生成labeledTrainData.csv文件，里面content是经过分词处理的
        # 直接返回review即可
        reviews = [str(line).strip().split() for line in review]

        return reviews, labels

    def _labelToIndex(self, labels, label2idx):
        """
        将标签转换成索引表示
        """
        labelIds = [label2idx[label] for label in labels]
        return labelIds

    def _wordToIndex(self, reviews, word2idx):
        """
        将词转换成索引
        """
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        return reviewIds

    def _genTrainEvalData(self, reviewIds, labelIds, word2idx, rate):
        """
        生成训练集和验证集
        """
        reviews = []
        # reviewIds是整个语料库的所有句子、每一个词的id，是一个矩阵
        # review是一个句子的词id，是一个向量
        for review in reviewIds:
            # 对词IDs进行长度截断，不足的补零向量（因为词ID是指已经在词库的词对应的ID，所有不存在未知词语，因此和UNK无光）
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))
        # 训练数据的索引，24000条 × 0.8 = 19200条，即训练数据索引从0～19200
        trainIndex = int(len(reviewIds) * rate)

        # 训练数据和训练数据的标签ID
        # 即语料库的词ID矩阵中获得0～19200条数据
        # !!!这里是有问题的，待优化
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labelIds[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labelIds[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self, reviews, labels):
        """
         i、生成词库、标签库文件 word2idx.json / label2idx.json
         ii、从词向量文件中获得词库每一个词的向量数据，放到self_wordEmbedding里
         1、获得所有内容里的分词 allWords
         2、过滤掉停用词 subWords
         3、获得词频数据 wordCound
         4、对词频数据进行倒序排，并去除低频词，最终获得 words
            去除低频词：能表示一句话特征的词，一般具备如下特征，在一句话里面出现次数少；但是在同类句子中出现频次高
                      因此在整个语料库范围内，去除低频词是必要的，可以减少无效计算
         5、调用函数_getWordEmbedding，从词向量文件中获取向量数据，vocab：词  embedding：向量
         6、向量数据放到self_wordEmbedding里
         7、标签数据去重、字典化之后，存入文件label2idx.json
         8、词数据（统计词频的时候已经去重），字典化之后存入word2idx.json
         """
        # reviews是二维集合，review是里面的一行，word是里面的一个元素
        # 这行代码的功能，就把reviews的二维集合转化为一维  可以这么看代码  word ((for review in reviews) for word in review)
        # 最里面的括号，通过reviews，定义了代码块内的review； 外层括号就应用了review，并定义了word.
        allWords = [word for review in reviews for word in review]

        # 去掉停用词, 其实就是word在allwords里面，但不再stopwordDict里
        subWords = [word for word in allWords if word not in self.stopWordDict]

        # 内置函数，统计词频
        wordCount = Counter(subWords)

        # 获得经过排序的词频数据
        # key是比较的值，lambda是一个隐函数，是固定写法，X代表是wordCount.items()的元素，x[1]就是词频,reverse是降序
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 【可优化】去除低频词  item[0]是词 / item[1]是词频   去除词频低于5的那些词，有利于提高特征区分度
        words = [item[0] for item in sortWordCount if item[1] >= 5]

        # 根据已经训练好的词向量，获得分析的词的向量
        # vocab 是词（按词频倒序排）  wordEmbedding 这个词的向量
        vocab, wordEmbedding = self._getWordEmbedding(words)
        # 词库里的词的向量数据
        self.wordEmbedding = wordEmbedding

        # 把词库转化为字典类型（字典类型查询速度会较快）
        word2idx = dict(zip(vocab, list(range(len(vocab)))))

        # set搭建无序不重复元素集，即去重
        uniqueLabel = list(set(labels))
        # 把标签数据转化为字典类型
        # 本来这里是把标签数据转为ID（以数量为上限）
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        # label2idx = dict(zip(uniqueLabel, uniqueLabel))
        self.labelList = list(range(len(uniqueLabel)))

        f_word2idx_json = os.path.join(basePath, 'data/word2idx.json')
        f_label2idx_json = os.path.join(basePath, 'data/label2idx.json')
        removeFileIfExists(f_word2idx_json)
        removeFileIfExists(f_label2idx_json)

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open(f_word2idx_json, "w", encoding="utf-8") as f:
            json.dump(word2idx, f, ensure_ascii=False)

        with open(f_label2idx_json, "w", encoding="utf-8") as f:
            json.dump(label2idx, f, ensure_ascii=False)

        return word2idx, label2idx

    def _getWordEmbedding(self, words):
        """
        1、读入已经生成的词向量文件
        2、从已有的词向量里面获取当前词的向量数据（不存在就警告并跳过）
        3、把当前词的词语（文字）和向量数据，存储到vocab / wordEmbdding，返回
        【理解说明】
        按照这段代码的逻辑，当前训练的词向量数据A的词，是要在另一个词向量数据B中存在
        由于输入的words是较高词频（低频词已经去除），而读入的词向量是整体语料库的词向量
        因此一般不会出现except的情况。
        """
        # 读取词向量文件，获取词向量数据
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(bin_word2Vec, binary=True)
        # vocab是词文件，里面是一个个单词，即语法库
        vocab = []
        # wordEmbedding是词向量，即用200维的数字数组来表示一个词
        wordEmbedding = []

        # 分词后不在词典内的词经常被标为<UNK>，处理为相同长度通常会在前或后补<PAD>
        # PAD：使用无损方法采用0向量   UNK(unknown):一般采用随机向量
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                # 从已经训练好的词向量数据中，获得这个单词的词向量
                vector = wordVec.wv[word]
                # 在语法库中添加词
                vocab.append(word)
                # 在词向量库中添加向量
                wordEmbedding.append(vector)
            except:
                log.warn(word + "不存在于词向量中")
        # 如果wordEmbedding的数据维度不一致，这里会抛出警告，可以修改
        # vec_size = 200 来保持维度一致
        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """
        if not stopWordPath is None:
            with open(stopWordPath, "r") as f:
                stopWords = f.read()
                stopWordList = stopWords.splitlines()
                # 将停用词用列表的形式生成，之后查找停用词时会比较快
                self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
        else:
            self.stopWordDict = {}

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化停用词：读入停用词文件，并把内容整理成dict，赋值给self.stopWordDict
        self._readStopWord(self._stopWordSource)

        # 从源数据文件中读取文本和标签数据
        reviews, labels = self._readData(self._dataSource)

        # 初始化词汇-索引映射表和词向量矩阵，生成<word2inx.json> <label2idx.json>
        # word2idx是词库， label2idx是标签库
        word2idx, label2idx = self._genVocabulary(reviews, labels)

        # 将标签和句子id化
        # labelIds是原始语料的标签ID， reivewIDs是原始语料每条预料的词ID
        # labels所有标签数据[标签id]，label2idx是[标签id：id], labelIds就是每条文本记录的标签ID[id,id,id,id...]
        # reviews是[[词, 词, ...][]]，word2idx是[词:词id]，reviewIds就是[[词id,词id,...],[词id,词id,...]...]
        labelIds = self._labelToIndex(labels, label2idx)
        reviewIds = self._wordToIndex(reviews, word2idx)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewIds, labelIds, word2idx,
                                                                                    self._rate)

        log.info('按照' + str(self._rate) + '的比例分配训练和测试数据完成')
        log.info('训练数据数量：' + str(len(trainLabels)))
        log.info('测试数据数量：' + str(len(evalLabels)))

        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels


"""
定义各类性能指标
"""


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_y, true_y):
    """
    计算二类和多类的准确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc


def binary_precision(pred_y, true_y, positive=1):
    """
    二类的精确率计算
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    pred_corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    """
    二类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    true_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            true_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    rec = corr / true_corr if true_corr > 0 else 0
    return rec


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    """
    二类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param beta: beta值
    :param positive: 正例的索引表示
    :return:
    """
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b


def multi_precision(pred_y, true_y, labels):
    """
    多类的精确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    prec = mean(precisions)
    return prec


def multi_recall(pred_y, true_y, labels):
    """
    多类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec


def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    """
    多类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :param beta: beta值
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    f_beta = mean(f_betas)
    return f_beta


def get_binary_metrics(pred_y, true_y, f_beta=1.0):
    """
    得到二分类的性能指标
    :param pred_y:
    :param true_y:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    precision = binary_precision(pred_y, true_y)
    f_beta = binary_f_beta(pred_y, true_y, f_beta)
    return acc, recall, precision, f_beta


def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
    """
    得到多分类的性能指标
    :param pred_y:
    :param true_y:
    :param labels:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    return acc, recall, precision, f_beta


# 输出batch数据集

def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


# 构建模型
class BiLSTMAttention(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        # tf.placeholder(dtype, shape, name)
        # dtype: 数据的类型，通常是int16,float32,float64
        # shape: 数据形状
        # name:  给变量定的名字
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失,这个参数将会参与基于交叉熵函数的loss值的计算
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):

            # 利用预训练的词向量初始化词嵌入矩阵
            # 【常见问题】会出现报转换失败的错误，从Linux服务器上复制word2Vec.bin可解决
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   self.embeddedWords, dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embeddedWords = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        outputs = tf.split(self.embeddedWords, 2, -1)

        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            H = outputs[0] + outputs[1]

            # 得到Attention的输出
            output = self.attention(H)
            outputSize = config.model.hiddenSizes[-1]

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                # 【常见问题】会发生module 'tensorflow.compat.v1' has no attribute 'contrib'错误
                # 解决办法是改成initializer=tf.glorot_uniform_initializer()
                # initializer=tf.contrib.layers.xavier_initializer()
                initializer=tf.glorot_uniform_initializer()
            )

            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            # 损失运算 用于测量两个张量之间或张量与0之间的误差。 这些可以用于测量回归任务中的网络的精确度，或用于正则化的目的（权重衰减）
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")

            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        # https: // blog.csdn.net / qq_35203425 / article / details / 79773459
        # 在多分类问题中，我们经常使用交叉熵作为损失函数，tensorflow中有四种交叉熵函数：
        # tf.nn.softmax_cross_entropy_with_logits()：softmax是对数字进行的归一化概率计算
        # tf.nn.sparse_softmax_cross_entropy_with_logits():输入的label数据和softmax不同
        # tf.nn.sigmoid_cross_entropy_with_logits():也叫Logistic函数，用于隐层神经元输出，取值范围为(0,1)，它可以将一个实数映射到(0,1)的区间
        # tf.nn.weighted_cross_entropy_with_logits(): 计算具有权重的sigmoid交叉熵
        with tf.name_scope("loss"):
            if config.numClasses == 1:
                # 如果是二分类，采用的是sigmoid交叉熵函数计算损失
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
                # 如果是多分类，采用的是sparse softmax交叉熵函数计算损失
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = config.model.hiddenSizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, config.sequenceLength])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, config.sequenceLength, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        # keep_prob: float类型，每个元素被保留下来的概率，设置神经元被选中的概率,在初始化时keep_prob是一个占位符
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output


"""
主程序

文件说明
【data/labeledTrainData.csv】：读入文件，是打标签的数据，包括content/label_id/label，文本已经经过清洗和分词
【data/word2Vec.bin】：输出文件，词向量文件，其中会生成word2Vec.tmp作为临时文件

业务流程
1、


"""

# 生成词向量文件
generateWord2VectorFile(file_labeled_train_data_csv, bin_word2Vec)

# 以上已经生成了labeledData.csv、labeledTrainData.csv、corpus.txt、word2Vec.bin
config = Config()
data = Dataset(config)
data.dataGen()

log.debug("train data shape: {}".format(data.trainReviews.shape))
log.debug("train label shape: {}".format(data.trainLabels.shape))
log.debug("eval data shape: {}".format(data.evalReviews.shape))

# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding
labelList = data.labelList

# 打开计算图
with tf.Graph().as_default():
    """
    ConfigProto参数：
        log_device_placement = True ： 可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,
            会在终端打印出各项操作是在哪个设备上运行的
        allow_soft_placement=True：允许tf自动选择一个存在并且可用的设备来运行操作。
            命令 "with tf.device('/cpu:0'):",允许手动设置操作运行的设备。如果手动设置的设备不存在或者不可用，就会导致tf程序等待或异常。
    """
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    """
    限制GPU资源使用：TF提供了两种控制GPU资源使用的方法:
    gpu_options.allow_growth = True  ：在运行过程中动态申请显存
    gpu_options.per_process_gpu_memory_fraction = 0.9  :  限制GPU的使用率
    """
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 打开会话
    with sess.as_default():
        # 定义一个模型算法，算法使用config对象为参数、并读入wordEmbedding作为数据
        lstm = BiLSTMAttention(config, wordEmbedding)

        # 以下是定义模型训练时候用的参数优化器：初始化优化计次、定义优化算法、生成优化器
        # 1、初始化优化计次：用以计算优化次数，从0开始，由apply_gradients在模型训练过程中，优化模型参数后+1
        # trainable一定要设置成False，这样梯度传播时就不会修改global_step
        # https://blog.csdn.net/shuibuzhaodeshiren/article/details/84669421
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 2、定义优化算法：Adam优化算法，一个寻找全局最优点的优化算法，引入了二次方梯度校正
        # 除了利用反向传播算法对权重和偏置项进行修正外，也在运行中不断修正学习率。
        # 根据其损失量学习自适应，损失量大则学习率大，进行修正的角度越大，损失量小，修正的幅度也小，学习率就小，
        # 但是不会超过自己所设定的最低学习率
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 根据模型的损失率数据，计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(lstm.loss)
        # 将梯度应用到变量下，生成训练优化器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        '''
        用summary绘制tensorBoard，即把模型优化过程数据用图形表现出来
        这里只写入了loss的数据
        '''
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                #  用来显示直方图信息，一般用来显示训练过程中变量的分布情况
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                # 用来显示标量信息，一般在画loss,accuary时会用到这个函数
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        # outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        outDir = os.path.join(basePath, 'data/model/summarys')
        log.debug("Summary data is writing to： {}".format(outDir))
        # 把lstm模型的loss的数据，写入到lossSummary
        lossSummary = tf.summary.scalar("loss", lstm.loss)

        summaryOp = tf.summary.merge_all()

        # 把train
        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        # savedModelPath = "../model/bilstm-atten/savedModel"
        savedModelPath = os.path.join(basePath, "data/model/savedModel")
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

        sess.run(tf.global_variables_initializer())


        def trainStep(batchX, batchY):
            """
            训练函数
            """
            feed_dict = {
                lstm.inputX: batchX,
                lstm.inputY: batchY,
                lstm.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions = sess.run(
                [trainOp, summaryOp, globalStep, lstm.loss, lstm.predictions],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()

            # 二分类的代码
            # if config.numClasses == 1:
            #     acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)

            # elif config.numClasses > 1:
            acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                          labels=labelList)

            trainSummaryWriter.add_summary(summary, step)

            return loss, acc, prec, recall, f_beta


        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                lstm.inputX: batchX,
                lstm.inputY: batchY,
                lstm.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, lstm.loss, lstm.predictions],
                feed_dict)

            if config.numClasses == 1:

                acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
            elif config.numClasses > 1:
                acc, precision, recall, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY, labels=labelList)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, precision, recall, f_beta


        for i in range(config.training.epoches):
            # 训练模型
            log.debug("epoches：{}{}".format(i, config.training.epoches))
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                log.debug("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta))

                if currentStep % config.training.evaluateEvery == 0:
                    log.debug("\nEvaluation:")

                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []

                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)

                    time_str = datetime.datetime.now().isoformat()
                    log.debug("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
                                                                                                             currentStep,
                                                                                                             mean(
                                                                                                                 losses),
                                                                                                             mean(accs),
                                                                                                             mean(
                                                                                                                 precisions),
                                                                                                             mean(
                                                                                                                 recalls),
                                                                                                             mean(
                                                                                                                 f_betas)))

                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, os.path.join(basePath, "data/model/"), global_step=currentStep)
                    log.debug("Saved model checkpoint to {}\n".format(path))

        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(lstm.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(lstm.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(lstm.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()

log.info('训练完成')
