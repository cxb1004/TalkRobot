"""

"""
import csv
import os
import sys

import pandas
import openpyxl

from pandas import DataFrame

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

from common.log import Log
from common.textSimilarity import CosSim
from common.utils import removeFileIfExists

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
    a = corpus_data[0]
    group.append(a)
    for b in corpus_data[1:]:
        idx = cosSim.getSimilarityIndex(a, b)
        if idx >= SIMILAR_IDX:
            group.append(b)
    return group


"""
1、读入tag_question.csv文件，读入questions列
2、读入knowledge_lib.csv，读入tagID/std_questions/sim_questions
3、如果knowledge_lib.csv存在
暂空
4、剩余的语料库中的数据，两两比较，相似度在0.8以上，组合
"""
#
# # 相似度指标
# SIMILAR_IDX = 0.8
#
# # 每个分类最少的问题数，如果低于这个问题，就废弃该分类
# MIN_QUESTION_IN_ONE_QUESTION = 3
#
# # 聚类分组ID默认起始值
# GROUP_IDX = 1
#
# # 已知聚类分析结果（和已有知识库匹配，归属于已有知识库下的聚类结果）  {'聚类名',['问题1'，'问题2'...]...}
# known_cluster_data = {}
#
# # 未知聚类分析结果（不和已有知识库匹配的，两两自我比较聚类的结果）  {'聚类名',['问题1'，'问题2'...]...}
# unknown_cluster_data = {}
#
# # 余弦相似比较器
# cosSim = CosSim()
#
# corpus_file = os.path.join(basePath, 'data/tag_question.csv')
# log.debug('读取文件{}:'.format(corpus_file))
# corpus_data = getCorpusData(corpus_file)
# log.debug('一共有{}条语料数据'.format(len(corpus_data)))
# # wordCount = Counter(corpus_data)
# # sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
# # duplicatedCount = list(item for item in sortWordCount if item[1]>1)
# # log.debug(duplicatedCount)
# corpus_data = list(set(corpus_data))
# log.debug('去重后有{}条语料数据'.format(len(corpus_data)))
#
# while len(corpus_data) > 0:
#     group = getGroup(corpus_data)
#
#     # 分组里语料数量大于3，这个分组为有效分组，加入到
#     if len(group) >= MIN_QUESTION_IN_ONE_QUESTION:
#         unknown_cluster_data[GROUP_IDX] = group
#         GROUP_IDX = GROUP_IDX + 1
#
#     # 分组数据无论是否最终采纳，都从原语料库数据中删除
#     if len(group) > 0:
#         removeItem(corpus_data, group)
#
# log.info('分析结果：')
# log.info(unknown_cluster_data)
unknown_cluster_data = {1:['a','b','c'],2:['d','e','f']}

result_file = os.path.join(basePath, 'data/result.xlsx')
removeFileIfExists(result_file)


def clusterToDataFrame(unknown_cluster_data):
    """
    转化字典类型到DataFrame类型
    :param unknown_cluster_data:聚类数据
    :return: DataFrame数据
    """
    df_data = []
    for tagId in unknown_cluster_data.keys():
        for questions in unknown_cluster_data.get(tagId):
            for question in questions:
                df_data.append([tagId, question])
    df = pandas.DataFrame(data=df_data)
    return df


if len(unknown_cluster_data) > 0:
    # 数据导出到xlsx里  第一列是GROUP_IDX,方便数据筛选  第二列是所有问题，第三列默认都是0，可以在excel里编辑为1，表示是标准问题

    unknown_df = clusterToDataFrame(unknown_cluster_data)

    # excel_writer: 文件路径或现有的ExcelWriter
    # sheet_name: 字符串, 默认“Sheet1”，将包含DataFrame的表的名称。
    # na_rep: 字符串, 默认‘ ’缺失数据表示方式
    # float_format: 字符串, 默认None
    # 格式化浮点数的字符串
    # columns: 序列, 可选 要编写的列
    # header: 布尔或字符串列表，默认为Ture。写出列名。如果给定字符串列表，则假定它是列名称的别名。
    # index: 布尔, 默认的Ture 写行名（索引）
    # index_label: 字符串或序列，默认为None。如果需要，可以使用索引列的列标签。如果没有给出，标题和索引为true，则使用索引名称。如果数据文件使用多索引，则需使用序列。
    # startrow: 左上角的单元格行来转储数据框
    # startcol: 左上角的单元格列转储数据帧
    # engine: 字符串, 默认没有。使用写引擎 - 您也可以通过选项io.excel.xlsx.writer，io.excel.xls.writer和io.excel.xlsm.writer进行设置。
    # merge_cells: 布尔, 默认为Ture
    # 编码生成的excel文件。 只有xlwt需要，其他编写者本地支持unicode。
    # inf_rep: 字符串, 默认“正”  无穷大的表示(在Excel中不存在无穷大的本地表示)
    # freeze_panes: 整数的元组(长度2)，默认为None。指定要冻结的基于1的最底部行和最右边的列
    unknown_df.to_excel(result_file,
                        'unknown_cluster_data',
                        na_rep='',
                        header=None,
                        encoding='utf-8'
                        )
