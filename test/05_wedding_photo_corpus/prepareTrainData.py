"""
数据说明
        {
            "id":
            "company_id":   # 公司ID
            "knowledge_id": # 问答ID
            "question":     # 问题文本
            "answer":       # 相似问题为null
            "industry_id":  # 行业ID
            "category_id":  # 分类ID
            "status": 1,    # 是否加入到行业库，软删除标志
            "parent_id":    # 关联字段，标准问题的knowledge_id
        }

例子
    标准问题
        {
            "id": 5732,
            "company_id": 72133216,
            "knowledge_id": "f08abd00e3e7b462e8cbd41376ef4d7a",
            "question": "请问你们店的地址在哪",
            "answer": "<p>\n\t亲亲，咱们全国都有店，您是在哪里拍呢？\n</p>",
            "industry_id": 67,
            "category_id": "18d34db8ba6ac758d057ce8d541d2399",
            "status": 2,
            "parent_id": "0"
        }
    相似问题
        {
            "id": 60243,
            "company_id": 72133216,
            "knowledge_id": "6a52058c58e1f4c789c690914d562f31",
            "question": "你这店地址在哪",
            "answer": "null",
            "industry_id": 67,
            "category_id": "18d34db8ba6ac758d057ce8d541d2399",
            "status": 1,
            "parent_id": "f08abd00e3e7b462e8cbd41376ef4d7a"
        }
"""
import csv
import json
import os
import sys

from bs4 import BeautifulSoup
import jieba
# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

from common.log import Log

log = Log()

source_file = os.path.join(basePath, 'data/response.txt')
tag_info_file = os.path.join(basePath, 'data/tag_info.csv')
tag_question_file = os.path.join(basePath, 'data/tag_question.csv')
labeled_train_data_csv = os.path.join(basePath, 'data/labeledTrainData.csv')


def segment(text):
    """
    去除标点符号，分词，用空格连接分词
    :param text: 文本
    :return: 去除html标签
    """
    # 对结果进行分词
    new_text = text
    word_list = jieba.cut(new_text)
    # 去除分词里的空格
    new_text = [word.lower() for word in word_list if word != ' ']
    # 使用空格拼接分词
    new_text = " ".join(new_text)
    return new_text

def dumpTagInfo(tag_desc_answer):
    with open(tag_info_file, 'w', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['tag_id', 'std_question', 'answer'])
        for tagKey in tag_desc_answer.keys():
            tagInfo = tag_desc_answer.get(tagKey)
            writer.writerow([tagKey, tagInfo[0], tagInfo[1]])


def dumpTagQuestions(tag_question):
    with open(tag_question_file, 'w', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['tag_id', 'questions'])
        for tagKey in tag_question.keys():
            questionList = tag_question.get(tagKey)
            for question in questionList:
                writer.writerow([tagKey, question])


def dumpLabeldedTrainData(tag_question):
    with open(labeled_train_data_csv, 'w', encoding='utf-8', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['content', 'label_id', 'label'])
        for tagKey in tag_question.keys():
            questionList = tag_question.get(tagKey)
            for question in questionList:
                question = clearText(question)
                question = segment(question)
                writer.writerow([question, tagKey, tagKey])


def clearText(text):
    # 去掉HTML标签
    beau = BeautifulSoup(text, "html.parser")
    # 去除HTML标
    new_text = beau.get_text()

    new_text = str(new_text).replace('\n', '')
    return new_text


def add_std_question_under_tag(tag_desc_answer, tag_question, data):
    # 新增标签信息，如果存在就覆盖
    tag = data['knowledge_id']
    question = clearText(data['question'])
    answer = clearText(data['answer'])
    tag_desc_answer[tag] = [question, answer]

    # 新增标准问题
    tag_question[tag] = [question]


def add_sim_question_under_tag(tag_desc_answer, tag_question, data):
    tag = data['parent_id']
    question = data['question']
    if tag_desc_answer.get(tag) is None:
        log.warn('标签[{}]缺失标准问题，补一条空问题！'.format(tag))
        tag_desc_answer[tag] = ['空问题', '空回答'+tag]
        # tag_question[tag] = ['空问题']

    if tag_question.get(tag) is not None:
        qList = tag_question.get(tag)
        qList.append(question)
        tag_question[tag] = qList
    else:
        tag_question[tag] = [question]


with open(source_file, 'r+', encoding='utf-8') as f:
    # 载入json内容
    jsonContent = json.load(f)
    # {}部分是字典类型 []部分是列表类型
    # log.info(type(jsonContent))
    # log.info(type(jsonContent['data']))
    # 获得data节点的内容
    all_data = jsonContent['data']
    cnt_all_data = len(all_data)

    # {tag: [desc, answer] , ...}
    tag_desc_answer = dict()
    # {tag: [question,question] , ...}
    tag_question = dict()
    for data in all_data:
        parent_id = data['parent_id']
        # 是否为标准问题
        is_std_question = 0
        if parent_id == '0':
            # parent_id=='0'没有上级问题，即为标准问题
            is_std_question = 1
        elif parent_id != '0':
            # parent_id<>'0'存在上级问题，即为相似问题
            is_std_question = 0

        # 问题文本
        q = data['question']

        if is_std_question == 1:
            if tag_desc_answer.get(data['knowledge_id']) is not None:
                q = tag_desc_answer.get(data['knowledge_id'])[0]
                if q == '空问题':
                    tag_desc_answer[data['knowledge_id']] = [data['question'], clearText(data['answer'])]
                    log.warn('标签[{}]问题修复！'.format(data['knowledge_id']))
                else:
                    log.warn('重复标签：  \"knowledge_id\": \"{}\"'.format(data['knowledge_id']))
            else:
                add_std_question_under_tag(tag_desc_answer, tag_question, data)
        else:
            add_sim_question_under_tag(tag_desc_answer, tag_question, data)

    dumpTagInfo(tag_desc_answer)
    dumpTagQuestions(tag_question)
    dumpLabeldedTrainData(tag_question)

cnt_tag = 0
with open(tag_info_file, 'r', encoding='utf-8') as csvFile:
    cnt_tag = len(csvFile.readlines())

log.info('===字数分布=================')

dist_wordLen = {}
all_len = 0
max_len = 0
cnt_questions = 0
with open(tag_question_file, 'r', encoding='utf-8') as csvFile:
    all_lines = csv.reader(csvFile)
    for lineData in all_lines:
        cnt_questions = cnt_questions + 1
        l = len(lineData[1])
        if l > max_len: max_len = l
        all_len = all_len + l
        if dist_wordLen.get(l) is None:
            dist_wordLen[l] = 1
        else:
            cnt = dist_wordLen.get(l) + 1
            dist_wordLen[l] = cnt
log.info('{}\t{}'.format('字数', '条数'))
for i in dist_wordLen.keys():
    log.info('{}\t{}'.format(i, dist_wordLen[i]))
log.info('=====================')

log.info('统计数据：')
log.info('一共有{}条数据'.format(cnt_all_data))
log.info('一共导入{}个标签'.format(cnt_tag))
log.info('一共导入{}个问题'.format(cnt_questions))
log.info('语料最长:{}字节'.format(max_len))
log.info('语料平均长:{}字节'.format(all_len / cnt_questions))
