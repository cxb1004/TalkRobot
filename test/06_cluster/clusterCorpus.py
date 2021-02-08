"""

"""
import csv
import os
import sys

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

from common.log import Log
from common.textSimilarity import CosSim

log = Log()


def getCorpusData(corpus_file):
    questions = []
    if not os.path.isfile(corpus_file):
        log.error('文件读取失败，请确认文件是否存在：{}'.format(corpus_file))
        return
    else:
        with open(corpus_file, 'r', encoding='utf-8') as csvFile:
            reader = csv.reader(csvFile)
            for line in reader:
                if reader.line_num != 1 and line[1].strip() != '':
                    questions.append(line[1])
    return questions


"""
1、读入tag_question.csv文件，读入questions列
2、读入knowledge_lib.csv，读入tagID/std_questions/sim_questions
3、如果knowledge_lib.csv存在
暂空
4、剩余的语料库中的数据，两两比较，相似度在0.8以上，组合
"""

# 相似度指标
SIMILAR_IDX = 0.8

# 每个分类最少的问题数，如果低于这个问题，就废弃该分类
MIN_QUESTION_IN_ONE_QUESTION = 3

# 聚类分组ID默认起始值
GROUP_IDX = 1

# 聚类分析结果  {'聚类名',['问题1'，'问题2'...]...}
cluster_data = {}

# 余弦相似比较器
cosSim = CosSim()

corpus_file = os.path.join(basePath, 'data/tag_question.csv')
log.debug('读取文件{}:'.format(corpus_file))
corpus_data = getCorpusData(corpus_file)
log.debug('一共有{}条语料数据'.format(len(corpus_data)))
# wordCount = Counter(corpus_data)
# sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
# duplicatedCount = list(item for item in sortWordCount if item[1]>1)
# log.debug(duplicatedCount)
corpus_data = list(set(corpus_data))
log.debug('去重后有{}条语料数据'.format(len(corpus_data)))


def removeItem(corpus_data, group):
    cnt_before = len(corpus_data)
    for item in group:
        try:
            corpus_data.remove(item)
        except:
            log.warn('元素不存在于语料库中：{}'.format(item))

    cnt_after = len(corpus_data)
    log.debug("分组数量是{}, 语料库从{}减少到{}".format(len(group), cnt_before, cnt_after))


def getGroup(corpus_data):
    group = []
    if len(corpus_data) == 1:
        corpus_data = {}
        return []
    else:
        a = corpus_data[0]
        group.append(a)
        for b in corpus_data[1:]:
            idx = cosSim.getSimilarityIndex(a, b)
            if idx >= SIMILAR_IDX:
                group.append(b)
    return group


while len(corpus_data) > 0:
    group = getGroup(corpus_data)

    if len(group) >= MIN_QUESTION_IN_ONE_QUESTION:
        cluster_data[GROUP_IDX] = group
        GROUP_IDX = GROUP_IDX + 1

    if len(group)>0:
        removeItem(corpus_data, group)

log.info('分析结果：')
log.info(corpus_data)
