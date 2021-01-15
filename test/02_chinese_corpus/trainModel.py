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

from common.log import Log

log = Log()

# 全局变量
# 定义标签和对应的ID，用于打标签
LABEL_ID_DICT = {'Auto': 0, 'Culture': 1, 'Economy': 2, 'Medicine': 3, 'Military': 4, 'Sports': 5}
# 反转标签的ID和标签值，用于查询
ID_LABEL_DICT = {v: k for k, v in LABEL_ID_DICT.items()}


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
    # TODO 标点符号的作用待讨论
    # 去除所有中英文标点符号：对于短文本来说，标点符号对于语义来说没有太大影响
    # global punctuation_en
    # for i in punctuation_str:
    #     new_text = text.replace(i, '')
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

    # corpus.txt文件如果已经存在，删除
    if os.path.isfile(file_corpus_txt):
        os.remove(file_corpus_txt)
    # word2Vec.bin文件如果已经存在，删除
    if os.path.isfile(_word2VecFile):
        os.remove(_word2VecFile)

    log.info("开始生成语料库文件：{}".format(file_corpus_txt))
    with open(_sourceFile, 'r', encoding='utf-8') as csvFile:
        with open(file_corpus_txt, 'w', encoding='utf-8', newline='') as corpusFile:
            # 读取labeledData.csv文件内容
            csvData = csv.reader(csvFile)
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

    log.info("语料库文件生成完毕：{}".format(file_corpus_txt))

    generateWord2Vector(_word2VecFile)


"""
主程序

文件说明
【data/labeledTrainData.csv】：读入文件，是打标签的数据，包括content/label_id/label，其中文本数据是需要经过清洗的
【data/corpus.txt】：输出文件，用于生成词向量文件
"""
# 读入数据：打过标签的数据文件 content / label_id / label
csv_labeled_data = os.path.join(basePath, 'data/labeledTrainData.csv')
bin_word2Vec = os.path.join(basePath, 'data/word2Vec.bin')

generateWord2VectorFile(csv_labeled_data, bin_word2Vec)
