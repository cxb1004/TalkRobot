"""
短句相似度判断
尝试使用多种算法对短句相似度进行判断，并对速度、消耗、精准度做综合评判
算法
1、simHash算法
2、基于词频的余弦相似度（TF-IDF）
3、官方difflib
4、编辑距离计算 Levenshtein 距离
5、杰卡德系数计算
6、词向量


"""
import difflib
import distance
import warnings
from collections import Counter
import re
import gensim
from sklearn.feature_extraction.text import CountVectorizer
import jieba
import jieba.analyse
import numpy as np
from scipy.linalg import norm
import os
import sys

warnings.filterwarnings("ignore")
# 以下部分代码可以保证在linux环境下任何目录都可以运行该文件
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


class SimHash(object):
    """
    simHash算法
    介绍：https://blog.csdn.net/ling620/article/details/95599549
         https://blog.csdn.net/ewen_lee/article/details/108443510
    """

    def simHash(self, content):
        seg = jieba.cut(content)
        # jieba.analyse.set_stop_words('stopword.txt')
        # jieba基于TF-IDF提取关键词
        keyWords = jieba.analyse.extract_tags("|".join(seg), topK=10, withWeight=True)

        keyList = []
        for feature, weight in keyWords:
            # print('weight: {}'.format(weight))
            # weight = math.ceil(weight)
            weight = int(weight)
            binstr = self.string_hash(feature)
            temp = []
            for c in binstr:
                if (c == '1'):
                    temp.append(weight)
                else:
                    temp.append(-weight)
            keyList.append(temp)
        listSum = np.sum(np.array(keyList), axis=0)
        if (keyList == []):
            return '00'
        simhash = ''
        for i in listSum:
            if (i > 0):
                simhash = simhash + '1'
            else:
                simhash = simhash + '0'

        return simhash

    def string_hash(self, source):
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            # print('strint_hash: %s, %s'%(source, x))

            return str(x)

    def getDistance(self, hashstr1, hashstr2):
        """
            计算两个simhash的汉明距离
        """
        length = 0
        # 【bug】字符长度差异，会产生下标越界的错误
        for index, char in enumerate(hashstr1):
            if char == hashstr2[index]:
                continue
            else:
                length += 1

        return length

    def getSimilarityIndex(self, std_str, cmp_str):
        """

        :param std_str: 标准的文本
        :param cmp_str: 比较的文本
        :return: 相似度指标
        """
        s1 = self.simHash(std_str)
        s2 = self.simHash(cmp_str)
        dist = self.getDistance(s1, s2)
        # print('SimHash result: {}'.format(dist))
        return dist


class CosSim(object):
    """
    基于词频的余弦相似度（TF-IDF）
    https://www.dazhuanlan.com/2020/02/28/5e57fc0ea7de2/
    """

    def getSimilarityIndex(self, input1, input2):
        str1 = jieba.lcut(input1)
        str2 = jieba.lcut(input2)
        co_str1 = (Counter(str1))
        co_str2 = (Counter(str2))
        p_str1 = []
        p_str2 = []
        for temp in set(str1 + str2):
            p_str1.append(co_str1[temp])
            p_str2.append(co_str2[temp])
        p_str1 = np.array(p_str1)
        p_str2 = np.array(p_str2)
        result = p_str1.dot(p_str2) / (np.sqrt(p_str1.dot(p_str1)) * np.sqrt(p_str2.dot(p_str2)))
        return round(result, 3)


class DiffLib(object):
    """
    Python difflib，difflib为python的标准库模块，无需安装，编辑距离的一种算法
    https://blog.csdn.net/Greepex/article/details/80493045?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~all~first_rank_v2~rank_v25-3-80493045.nonecase&utm_term=%E4%B8%AD%E6%96%87%E7%9F%AD%E6%96%87%E6%9C%AC%E7%9B%B8%E4%BC%BC%E5%BA%A6&spm=1000.2123.3001.4430
    """

    def getSimilarityIndex(self, str1, str2):
        diff_result = difflib.SequenceMatcher(None, str1, str2).ratio()
        return diff_result


class Levenshtein(object):
    def getSimilarityIndex(self, s1, s2):
        return distance.levenshtein(s1, s2)


class Jaccard(object):
    def getSimilarityIndex(self, s1, s2):
        # 将字中间加入空格
        s1, s2 = self.add_space(s1), self.add_space(s2)
        # 转化为TF矩阵
        cv = CountVectorizer(tokenizer=lambda s: s.split())
        corpus = [s1, s2]
        vectors = cv.fit_transform(corpus).toarray()
        # 求交集
        numerator = np.sum(np.min(vectors, axis=0))
        # 求并集
        denominator = np.sum(np.max(vectors, axis=0))
        # 计算杰卡德系数
        return 1.0 * numerator / denominator

    def add_space(self, s):
        return ' '.join(list(s))


class VecSim(object):
    model = None

    def __init__(self):
        model_file = 'D:/PycharmProjects/EntityRecognizion/53data/word2Vec.bin'
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

    def sentence_vector(self, s):
        if s is '':
            return 0
        s = re.sub("？", "", s)
        s = re.sub("z", "", s)
        s = re.sub("a", "", s)
        s = re.sub("i", "", s)
        words = jieba.lcut(s)
        v = np.zeros(200)
        for word in words:
            v += self.model[word]
        v /= len(words)
        return v

    def getSimilarityIndex(self, s1, s2):
        try:
            v1, v2 = self.sentence_vector(s1), self.sentence_vector(s2)
            result = np.dot(v1, v2) / (norm(v1) * norm(v2))
        except:
            return 0
        return result


# 测试用代码
# std_str0 = '请问这件商品的价格是多少钱？'
# cmp_strArray = ['请问这件商品的价格是多少？', '这件商品的价格是？', '这个多少钱？', '恩', '我买了', '这个商品有什么特点？', 'zai', '有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗']
# if __name__ == '__main__':
#     simhash = SimHash()
#     sinHashIdx = simhash.getSimilarityIndex(std_str0, std_str0)
#     print('SimHash result: {}： {}'.format(sinHashIdx, std_str0))
#     for cmp_str in cmp_strArray:
#         sinHashIdx = simhash.getSimilarityIndex(cmp_str, std_str0)
#         print('SimHash result: {}： {}'.format(sinHashIdx, cmp_str))
#
#     print('')
#     print('')
#     cosSim = CosSim()
#     cosSimIdx = cosSim.getSimilarityIndex(std_str0, std_str0)
#     print('CosSim result: {}： {}'.format(cosSimIdx, std_str0))
#     for cmp_str in cmp_strArray:
#         cosSimIdx = cosSim.getSimilarityIndex(std_str0, cmp_str)
#         print('CosSim result: {}： {}'.format(cosSimIdx, cmp_str))
#
#     print('')
#     print('')
#     diffLib = DiffLib()
#     diffLibIdx = diffLib.getSimilarityIndex(std_str0, std_str0)
#     print('DiffLib result: {}： {}'.format(diffLibIdx, std_str0))
#     for cmp_str in cmp_strArray:
#         diffLibIdx = diffLib.getSimilarityIndex(std_str0, cmp_str)
#         print('DiffLib result: {}： {}'.format(diffLibIdx, cmp_str))
#
#     print('')
#     print('')
#     levenshtein = Levenshtein()
#     levenshteinIdx = levenshtein.getSimilarityIndex(std_str0, std_str0)
#     print('Levenshtein result: {}： {}'.format(levenshteinIdx, std_str0))
#     for cmp_str in cmp_strArray:
#         levenshteinIdx = levenshtein.getSimilarityIndex(std_str0, cmp_str)
#         print('Levenshtein result: {}： {}'.format(levenshteinIdx, cmp_str))
#
#     print('')
#     print('')
#     jaccard = Jaccard()
#     jaccardIdx = jaccard.getSimilarityIndex(std_str0, std_str0)
#     print('Jaccard result: {}： {}'.format(jaccardIdx, std_str0))
#     for cmp_str in cmp_strArray:
#         jaccardIdx = jaccard.getSimilarityIndex(std_str0, cmp_str)
#         print('Jaccard result: {}： {}'.format(jaccardIdx, cmp_str))
#
#     print('')
#     print('')
#     vecSim = VecSim()
#     vecSimIdx = vecSim.getSimilarityIndex(std_str0, std_str0)
#     print('VecSim result: {}： {}'.format(vecSimIdx, std_str0))
#     for cmp_str in cmp_strArray:
#         vecSimIdx = vecSim.getSimilarityIndex(std_str0, cmp_str)
#         print('VecSim result: {}： {}'.format(vecSimIdx, cmp_str))


"""
SimHash result: 0： 请问这件商品的价格是多少钱？
SimHash result: 0： 请问这件商品的价格是多少？
SimHash result: 17： 这件商品的价格是？
SimHash result: 13： 这个多少钱？
SimHash result: 0： 恩
SimHash result: 0： 我买了
SimHash result: 20： 这个商品有什么特点？
SimHash result: 34： zai
SimHash result: 34： 有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗
【分析】按照理论，数值越小，相似度越高。但是错误率高，相似度也不准。 

CosSim result: 1.0： 请问这件商品的价格是多少钱？
CosSim result: 0.943： 请问这件商品的价格是多少？
CosSim result: 0.816： 这件商品的价格是？
CosSim result: 0.5： 这个多少钱？
CosSim result: 0.0： 恩
CosSim result: 0.0： 我买了
CosSim result: 0.272： 这个商品有什么特点？
CosSim result: 0.0： zai
CosSim result: 0.0： 有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗
【分析】以0.5作为标阀值分类正确，并且相似度数值也较合理

DiffLib result: 1.0： 请问这件商品的价格是多少钱？
DiffLib result: 0.9629629629629629： 请问这件商品的价格是多少？
DiffLib result: 0.782608695652174： 这件商品的价格是？
DiffLib result: 0.5： 这个多少钱？
DiffLib result: 0.0： 恩
DiffLib result: 0.0： 我买了
DiffLib result: 0.3333333333333333： 这个商品有什么特点？
DiffLib result: 0.0： zai
DiffLib result: 0.04878048780487805： 有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗
【分析】以0.5作为阀值，分类正确，并且相似度数值也较合理

Levenshtein result: 0： 请问这件商品的价格是多少钱？
Levenshtein result: 1： 请问这件商品的价格是多少？
Levenshtein result: 5： 这件商品的价格是？
Levenshtein result: 9： 这个多少钱？
Levenshtein result: 14： 恩
Levenshtein result: 14： 我买了
Levenshtein result: 10： 这个商品有什么特点？
Levenshtein result: 14： zai
Levenshtein result: 26： 有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗
【分析】以10作为标准，分类正确，并且相似度数值也较合理

Jaccard result: 1.0： 请问这件商品的价格是多少钱？
Jaccard result: 0.9285714285714286： 请问这件商品的价格是多少？
Jaccard result: 0.6428571428571429： 这件商品的价格是？
Jaccard result: 0.3333333333333333： 这个多少钱？
Jaccard result: 0.0： 恩
Jaccard result: 0.0： 我买了
Jaccard result: 0.2： 这个商品有什么特点？
Jaccard result: 0.0： zai
Jaccard result: 0.025： 有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗
【分析】以0.3作为标准，分类正确，并且相似度数值也较合理

VecSim result: 1.0000000000000004： 请问这件商品的价格是多少钱？
VecSim result: 0.9814942154763796： 请问这件商品的价格是多少？
VecSim result: 0.8986491557467426： 这件商品的价格是？
VecSim result: 0.7759321598444318： 这个多少钱？
VecSim result: 0.34691092720797545： 恩
VecSim result: 0.7017638259543683： 我买了
VecSim result: 0.7981520735199975： 这个商品有什么特点？
VecSim result: 0： zai
VecSim result: 0.5987027396270072： 有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗有钱吗
【分析】分类有错误，相似度较合理

Process finished with exit code 0
--------------------------------------------------------------------------------------
以上是运行结果和分析
除去分类错误，剩下的算法有CosSim / DiffLib / Levenshtein / Jaccard
其中余弦算法是主流算法，就采用这个。
词向量理论上精准度较高，优势是基于训练的词向量文件，因此在特定业务中准确率会较高
但是对比较的文本需要进行清洗，限定在词向量的词库范围内，对相似度比较这个单一功能来说，会有限制。

"""
