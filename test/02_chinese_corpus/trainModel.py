"""

"""
import os
import sys

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

import csv
import re
from gensim.models import word2vec
import gensim
from bs4 import BeautifulSoup
import jieba
import pandas as pd
import numpy as np
from collections import Counter
import json
from zhon.hanzi import punctuation as punctuation
import string

from common.log import Log

log = Log()

# 全局变量
# 定义标签和对应的ID，用于打标签
LABEL_ID_DICT = {'Auto': 0, 'Culture': 1, 'Economy': 2, 'Medicine': 3, 'Military': 4, 'Sports': 5}
# 反转标签的ID和标签值，用于查询
ID_LABEL_DICT = {v: k for k, v in LABEL_ID_DICT.items()}

# 读入数据：打过标签的数据文件 content / label_id / label
csv_labeled_data = os.path.join(basePath, 'data/labeledData.csv')

bin_word2Vec = os.path.join(basePath, 'data/word2Vec.bin')

# 需要过滤的标点符号，出于语义考虑，保留问号
fileter_punctuation = (string.punctuation + punctuation) \
    .replace('?', '') \
    .replace('？', '')

def getCutWordsWithClean(text):
    """
    对文本进行清洗
    1、去除html标签
    2、去除[img]
    3、去除自定义的表情符号 {53c_min#xx#}
    4、替换部分标点符号，考虑到中文标点符号对于语义的影响，不清除标点符号
    :param text: 文本
    :return: 去除html标签
    """
    beau = BeautifulSoup(text, "html.parser")
    # 去除HTML标
    new_text = beau.get_text()

    # 去除{53c_min#xx#}
    pattern = re.compile(r'({53c_min#)(.*)(#})')
    new_text = pattern.sub(r'', new_text)

    # # 去除[img]...[/img]
    pattern = re.compile(r'(\[img\])(.*)(\[\/img\])')
    new_text = pattern.sub(r'', new_text)

    # 涉及语义的英文字符替换成中文的
    new_text = new_text.replace('?', '？')
    new_text = new_text.replace('!', '！')

    return new_text


def generateWord2Vector(_word2VecFile):
    """
    根据语料库文件生成词向量文件
    :param file_corpus_txt:
    :return:
    """
    log.info("开始生成词向量文件：{}".format(bin_word2Vec))
    # 根据word2Vec.bin的位置，生成corpus.txt的位置
    path_word2Vec = os.path.dirname(os.path.abspath(_word2VecFile))
    file_corpus_txt = os.path.join(path_word2Vec, 'corpus.txt')
    if not os.path.isfile(file_corpus_txt):
        log.error('文件不存在，请确认：{}'.format(file_corpus_txt))
        raise FileNotFoundError('文件不存在，请确认：{}'.format(file_corpus_txt))

    # word2Vec.bin文件如果已经存在，删除
    if os.path.isfile(_word2VecFile):
        os.remove(_word2VecFile)

    sentences = word2vec.LineSentence(file_corpus_txt)
    vec_size = 100
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


def segment(text):
    """
    去除标点符号，分词，用空格连接分词
    :param text: 文本
    :return: 去除html标签
    """
    # 去除所有中英文标点符号：对于短文本来说，标点符号对于语义来说没有太大影响，保留了问号
    global fileter_punctuation
    for i in fileter_punctuation:
        new_text = text.replace(i, '')
    # 对结果进行分词
    new_text = text
    word_list = jieba.cut(new_text)
    # 去除分词里的空格
    new_text = [word.lower() for word in word_list if word != ' ']
    # 使用空格拼接分词
    new_text = " ".join(new_text)
    return new_text


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
        return 9

    # 根据word2Vec.bin的位置，生成corpus.txt的位置
    path_word2Vec = os.path.dirname(os.path.abspath(_word2VecFile))
    file_corpus_txt = os.path.join(path_word2Vec, 'corpus.txt')
    file_labeled_train_data_csv = os.path.join(path_word2Vec, 'labeledTrainData.csv')

    # corpus.txt文件如果已经存在，删除
    if os.path.isfile(file_corpus_txt):
        os.remove(file_corpus_txt)
    # word2Vec.bin文件如果已经存在，删除
    if os.path.isfile(_word2VecFile):
        os.remove(_word2VecFile)

    log.info("开始生成语料库文件：{}".format(file_corpus_txt))
    with open(_sourceFile, 'r', encoding='utf-8') as csvFile:
        with open(file_corpus_txt, 'w', encoding='utf-8', newline='') as corpusFile:
            with open(file_labeled_train_data_csv, 'w', encoding='utf-8', newline='') as trainDataCSV:
                # 读取labeledData.csv文件内容
                csvData = csv.reader(csvFile)
                csvTrainData = csv.writer(trainDataCSV)
                csvTrainData.writerow(['content', 'label_id', 'label'])

                for line in csvData:
                    # 绕过第一行的标题
                    if not csvData.line_num == 1:
                        content = line[0]
                        label_id = line[1]
                        label_txt = line[2]
                        # 如果纯粹是空行，跳过
                        if not content == '\n':
                            # 对文本数据进行清洗
                            write_line = getCutWordsWithClean(content)
                            # 对文本数据进行中文分词
                            write_line = segment(write_line)
                            # 写入文件
                            corpusFile.write(write_line + '\n')
                            csvTrainData.writerow([write_line, label_id, label_txt])

    log.info("语料库文件生成完毕：{}".format(file_corpus_txt))

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
    # 学习速率参数 （无需改动）
    # 用于Adam优化器：Adam即Adaptive Moment Estimation（自适应矩估计），是一个寻找全局最优点的优化算法，引入了二次梯度校正
    # 默认的参数就是0.001
    # 根据其损失量学习自适应，损失量大则学习率大，进行修正的角度越大，损失量小，修正的幅度也小，学习率就小，但是不会超过自己所设定的学习率
    learningRate = 0.001


class ModelConfig(object):
    """
    模型的配置
    """
    embeddingSize = 200
    # ？？ LSTM结构的神经元个数
    hiddenSizes = [256, 128]
    # 过拟合参数
    # 过拟合的意思就是在划分特征的时候，过度贴近于某几个特征点，从而导致向量机变得复杂
    # https://blog.csdn.net/star_of_science/article/details/104245506
    dropoutKeepProb = 0.5
    # ？？
    l2RegLambda = 0.0


class Config(object):
    """
    主要配置参数，包括运行
    """
    sequenceLength = 200  # 取了所有序列长度的均值
    batchSize = 128

    # 【待重构】
    # 原始数据源
    # dataSource = os.path.join(basePath, "data/labeledTrain.csv")
    dataSource = csv_labeled_data

    # 【待重构】
    # 暂时先不考虑停用词
    # stopWordSource = os.path.join(basePath, "data/english")
    stopWordSource = None

    # 【待重构】
    # 这里我们用到的是多分类，不是二分类，参考csv文件中有10个分类，因此设置为10
    # numClasses = 11
    numClasses = len(LABEL_ID_DICT)

    # 训练集的比例，即80%用于训练，20%用于测试
    rate = 0.8

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
        labels = df["label_id"].tolist()
        review = df["content"].tolist()
        # 【已变更】原有功能是每一行去除前后空格之后，按空格划分，即对英文进行分词操作
        # 而中文的分词需要特殊处理，所以这里要前置处理，在生成csv文件的时候，就把分词结果写入到content里面
        # 因此在生成corpus.txt文件的时候，同时生成labeledTrainData.csv文件，里面content是经过分词处理的
        # 直接返回review即可
        # reviews = [line.strip().split() for line in review]

        return review, labels

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

    def _genTrainEvalData(self, x, y, word2idx, rate):
        """
        生成训练集和验证集
        """
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))

        trainIndex = int(len(x) * rate)

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(y[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(y[trainIndex:], dtype="float32")

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
        # label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        label2idx = dict(zip(uniqueLabel, uniqueLabel))
        self.labelList = list(range(len(uniqueLabel)))

        f_word2idx_json = os.path.join(basePath, 'data/word2idx.json')
        f_label2idx_json = os.path.join(basePath, 'data/label2idx.json')
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
        f_word2Vec_bin = os.path.join(basePath, 'data/word2Vec.bin')
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(f_word2Vec_bin, binary=True)
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

        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labels)

        # 将标签和句子id化
        # labels所有标签数据[标签]，label2idx是[标签：id], labelIds就是每条文本记录的标签ID[id]
        # reviews是[[词, 词, ...][]]，word2idx是[词:id]，reviewIds就是[[id,id,...],[]]
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
主程序

文件说明
【data/labeledData.csv】：读入文件，是打标签的数据，包括content/label_id/label，其中文本数据是需要经过清洗的
【data/corpus.txt】：输出文件，用于生成词向量文件
"""

# 生成词向量文件
generateWord2VectorFile(csv_labeled_data, bin_word2Vec)

# # Todo 以下是新代码
# config = Config()
#
# data = Dataset(config)
# data.dataGen()
#
# log.debug("train data shape: {}".format(data.trainReviews.shape))
# log.debug("train label shape: {}".format(data.trainLabels.shape))
# log.debug("eval data shape: {}".format(data.evalReviews.shape))
#
# # 训练模型
#
# # 生成训练集和验证集
# trainReviews = data.trainReviews
# trainLabels = data.trainLabels
# evalReviews = data.evalReviews
# evalLabels = data.evalLabels
#
# wordEmbedding = data.wordEmbedding
# labelList = data.labelList
