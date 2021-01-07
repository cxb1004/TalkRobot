"""
导入系统类库并设定运行根目录
"""
import datetime
import os
import sys

# 当前目录
curPath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(curPath)

"""
导入第三方类库
"""
import shutil
import gensim
import pandas as pd
import numpy as np
import json
import tensorflow.compat.v1 as tf
from collections import Counter
from gensim.models import word2vec

"""
导入我方项目类库
"""
from common.log import Log
from common.db import Database

"""
全局变量
"""
# data目录
dataPath = os.path.join(curPath, 'data')
# corpus文件
corpus_file = os.path.join(dataPath, 'corpus.txt')
# 日志
log = Log()

# 获取标签数据，并生成字典
label_dict = {}
if label_dict is None:
    # 从数据表读取标签文件
    db = Database()
    sql = 'select id,label from  robot_label where status=0 order by id desc;'
    try:
        result_labels = db.query(sql, None)
        log.info('获取成功！')
    except Exception as e:
        log.error('数据库操作失败：{}'.format(sql))
        log.error(e)
        exit(999)
    finally:
        db.close()
    for d in result_labels:
        label_dict[d[0]] = d[1]

"""
子程序
"""


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
    主配置函数
    """
    # 取了所有序列长度的均值
    sequenceLength = 200

    # batch：使用训练集中的一小部分样本对模型权重进行反向传播用于更新模型参数，称为一批数据
    batchSize = 128

    # 当日的整理出来的打过标签的数据
    dataSource = os.path.join(dataPath, 'labeledTrainData.csv')
    if not os.path.exists(dataSource):
        log.error('训练数据不存在，请检查：' + str(dataSource))
        exit(999)

    # 停用词文件：以分行的形式
    stopWordSource = os.path.join(dataPath, "stopword")

    # 二分类设置为1，多分类设置为类别的数目
    global label_dict
    numClasses = len(label_dict)

    # 训练集的比例, 这个比例是把90%的数据来用于训练模型，10%的数据来验证模型
    rate = 0.9

    # 训练的配置
    training = TrainingConfig()

    # 模型的配置
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
        # 设置停用词文件
        self._stopWordSource = config.stopWordSource
        # 设置序列长度200
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        # 设置特征数量200
        self._embeddingSize = config.model.embeddingSize
        # 设置批处理数量128（用于多次运行，修正模型参数）
        self._batchSize = config.batchSize
        # 设置训练集的比例，即80%的数据用来进行训练，20%数据用于测试
        self._rate = config.rate

        # 初始化停用词字典，在_readStopWord()进行初始化
        self._stopWordDict = {}

        # 训练集 review指的是文本内容，label是标签
        self.trainReviews = []
        self.trainLabels = []

        # 测试集
        self.evalReviews = []
        self.evalLabels = []

        # 训练集中的词向量
        self.wordEmbedding = None

        # 标签列表
        self.labelList = []

    def _readData(self, filePath):
        """
        从labeledTrainData.csv文件中读取数据集，不包含第一行
        """
        # 用pandas读取csv数据文件
        df = pd.read_csv(filePath, encoding='utf-8')

        # 这里是把标签从原始数据文件中提取出来
        # 数据文件中：
        # 第一列：文本内容
        # 第二列：标签ID

        # 读取label列
        labels = df["label_id"].tolist()
        # 读取content列
        review = df["content"].tolist()

        # 在上述review列表基础上，把每个元素再按空格进行切分
        # 数据清洗的时候，已经进行了中文分词，并以空格拼接，因此这里直接用空格进行切分即可
        reviews = [line.strip().split() for line in review]

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
        allWords = [_word for _review in reviews for _word in _review]

        # 去掉停用词, 其实就是word在allwords里面，但不再stopwordDict里
        subWords = [_word for _word in allWords if _word not in self.stopWordDict]

        # 内置函数，统计词频
        wordCount = Counter(subWords)

        # 获得经过排序的词频数据
        # key是比较的值，lambda是一个隐函数，是固定写法，X代表是wordCount.items()的元素，x[1]就是词频,reverse是降序
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词  item[0]是词 / item[1]是词频   去除词频低于5的那些词，有利于提高特征区分度
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
        self.labelList = list(uniqueLabel)

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open(word2idx_json_file, "w", encoding="utf-8") as f:
            json.dump(word2idx, f, ensure_ascii=False)

        # 把标签-索引映射表保存为json数据
        with open(label2idx_json_file, "w", encoding="utf-8") as f:
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
        global word2vec_file
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)

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
        # 返回的是已经经过训练的词、以及词向量
        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):
        """
        读取停用词文件
        """
        with open(stopWordPath, "r") as f:
            # 读取停用词文件的文件内容（整体）
            stopWords = f.read()
            # 把数据分行，返回行数据列表
            stopWordList = stopWords.splitlines()
            # 将停用词列表用字典的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def dataGen(self):
        """
        初始化训练集和验证集
        """
        log.info('开始生成训练集和验证集...')

        # 初始化停用词：读入停用词文件，并把内容整理成dict，赋值给self.stopWordDict
        self._readStopWord(self._stopWordSource)
        log.info('停用词读取完毕')

        # 从源数据文件中读取文本和标签数据
        reviews, labels = self._readData(self._dataSource)
        log.info('从源数据文件中读取文本和标签数据完毕')

        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labels)
        log.info('解析词索引和标签索引完毕')

        # 将标签和句子id化
        # labels所有标签数据[标签]，label2idx是[标签：id], labelIds就是每条文本记录的标签ID[id]
        # reviews是[[词, 词, ...][]]，word2idx是[词:id]，reviewIds就是[[id,id,...],[]]
        labelIds = self._labelToIndex(labels, label2idx)
        log.info('标签数据转化为标签ID完毕')
        reviewIds = self._wordToIndex(reviews, word2idx)
        log.info('文本数据转化为词ID完毕')

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


def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    采用生成器的形式向模型输入batch数据集，（生成器可以避免将所有的数据加入到内存中）
    输入：evalReviews, evalLabels
        x: 训练数据的文本
        y: 训练数据的标签
        batchSize:
    输出：

    """
    # arange返回一个有终点和起点的固定步长的排列，如[0,1,2,...len(x)-1]
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    numBatches = len(x) // batchSize

    # cxb更改： 数据不足128条，也要进行运算
    # if (len(x) > 0) & numBatches == 0:
    #     numBatches = 1

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


class BiLSTMAttention(object):
    """
    Text CNN 用于文本分类
    """

    def __init__(self, config, wordEmbedding):
        """

        :param config:训练模型时需要用到的参数
        :param wordEmbedding: 词向量数据
        """

        # 定义模型的输入
        # placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
        # 等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据
        # 参数有三个：
        #   dtype：数据类型。
        #   shape：数据形状。默认是None，就是一维值，也可以是多维，比如[None, config.sequenceLength]表示行不定，列长200的二维数据
        #   name：名称
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")
        # 过拟合参数
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):

            # 利用预训练的词向量初始化词嵌入矩阵
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
                # cxb: 替换
                # initializer=tf.contrib.layers.xavier_initializer())
                initializer=tf.truncated_normal_initializer(stddev=0.1))

            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")

            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):

            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
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
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output


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


"""
主程序
"""

# 同级的data目录作为数据文件的根目录
dataPath = os.path.join(curPath, 'data')
# 模型文件生成目录
trainPath = os.path.join(dataPath, 'train')
modelPath = os.path.join(trainPath, 'model')

# 输入文件：语料库文件，读取，用以生成词向量文件
corpus_file = os.path.join(dataPath, 'corpus.txt')
# 输出文件：词向量文件
word2vec_file = os.path.join(trainPath, 'word2Vec.bin')
# 输出文件：词库word2idx.json文件路径
word2idx_json_file = os.path.join(trainPath, 'word2idx.json')
# 输出文件：标签库label2idx.json文件路径
label2idx_json_file = os.path.join(trainPath, 'label2idx.json')

# 重建模型数据目录
log.info('重建模型目录')
if not os.path.exists(trainPath):
    os.mkdir(trainPath)
else:
    shutil.rmtree(trainPath)
    os.mkdir(trainPath)

"""
生成词向量文件
"""
log.info('生成词向量文件...')
sentences = word2vec.LineSentence(corpus_file)
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
model = gensim.models.Word2Vec(sentences,
                               size=vec_size,
                               window=vec_window,
                               min_count=vec_min_count,
                               sg=vec_sg,
                               iter=vec_iter)
log.info('生成中...')
model.wv.save_word2vec_format(word2vec_file, binary=True)
log.info('生成完毕')

log.info('配置训练参数...')
# 设置配置参数，包括运行配置、训练配置、模型配置等
config = Config()
log.info('配置训练参数完毕')

# 读入配置
data = Dataset(config)
log.info('准备开始生成训练/测试数据...')

# 生成训练数据和测试数据
data.dataGen()
log.info('模型训练所需的数据文件生成完毕')

log.info('训练/测试数据情况如下：')
log.debug("train data shape: {}".format(data.trainReviews.shape))
log.debug("train label shape: {}".format(data.trainLabels.shape))
log.debug("eval data shape: {}".format(data.evalReviews.shape))
log.debug("eval label shape: {}".format(data.evalLabels.shape))

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding
labelList = data.labelList

# 检查模型输出目录是否存在
if not os.path.exists(modelPath):
    os.makedirs(modelPath)

log.info('准备完毕，开始训练模型...')
# https://www.jianshu.com/p/ec261a65e3c9
# Tensorflow的设计理念称之为计算流图
# 在编写程序时，首先构筑整个系统的graph，代码并不会直接生效，graph为静态的，类似于docker中的镜像。
# 在实际的运行时，启动一个session，程序才会真正的运行。
# 这样做的好处就是：避免反复地切换底层程序实际运行的上下文，tensorflow帮你优化整个系统的代码。
# tensorflow通过计算流图的方式，帮你优化整个session需要执行的代码，还是很有优势的。
with tf.Graph().as_default():
    # 我们目前的环境不支持GPU，因此下面的参数意义仅作了解
    # allow_soft_placement: 如果指定的设备不存在，允许TF自动分配设备。 必须要，TensorFlow2已经不区分GPU和CPU，如果GPU不存在，就使用CPU设备
    # log_device_placement:是否打印设备分配日志
    # allow_growth：分配器将不会指定所有的GPU内存，而是根据需求增长
    # per_process_gpu_memory_fraction: 每个GPU上使用显存的比例，均匀作用于每个GPU
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
    # 用上面的配置初始化session
    sess = tf.Session(config=session_conf)

    # 创建一个默认会话，当上下文管理器退出时会话没有关闭，还可以通过调用会话进行run()和eval()操作
    with sess.as_default():
        # ??
        lstm = BiLSTMAttention(config, wordEmbedding)

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        # 根据其损失量学习自适应，损失量大则学习率大，进行修正的角度越大，损失量小，修正的幅度也小，学习率就小，但是不会超过自己所设定的学习率
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(lstm.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        # outDir = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), "data/summarys"))
        outDir = os.path.join(trainPath, "summarys")
        log.info("训练和评估结果数据目录创建： {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", lstm.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 设置第一种保存模型的方式：保存结果为 variables目录（存放模型参数）和一个pb文件（模型本身）
        #  max_to_keep 参数表示要保留的最近检查点文件的最大数量，
        #  创建新文件时，将删除旧文件，默认为 5（即保留最近的 5 个检查点文件）
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        savedModelPath = modelPath
        if os.path.exists(savedModelPath):
            shutil.rmtree(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)
        log.info('第一种种保存模型方式设置完毕，保存路径为:' + str(savedModelPath))

        sess.run(tf.global_variables_initializer())
        log.info('初始化模型参数完毕')


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

            if config.numClasses == 1:
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)


            elif config.numClasses > 1:
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


        log.info('训练中...')
        for i in range(config.training.epoches):
            # 训练模型
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                log.debug(
                    "train:: epoches:{}, step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(i,
                                                                                                                    currentStep,
                                                                                                                    loss,
                                                                                                                    acc,
                                                                                                                    recall,
                                                                                                                    prec,
                                                                                                                    f_beta))
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
                # 以下是第二种模型保存方案，如果不需要可以删除（预测的时候加载的是这个模型，这里如果注释，预测代码需要改）
                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    my_model_path = os.path.join(modelPath, 'model/Bi-LSTM-atten/model/my-model')
                    if not os.path.exists(my_model_path):
                        os.makedirs(my_model_path)

                    path = saver.save(sess, my_model_path, global_step=currentStep)
                    log.info("第二种模型保存方式设定，保存目录为： {}\n".format(path))

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
log.info('模型训练完成！')
