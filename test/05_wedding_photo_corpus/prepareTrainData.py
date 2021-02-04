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


def add_std_question_under_tag(tag_question, tag, q):
    """
    把问题加入到标签下的问题列表中
    1、先检查tag_question是否存在标签，不存在就做插入操作
    2、如果存在，去除问题列表对象，查看q是否已经存在于列表，存在则抛出警告，不存在就插入操作
    :param tag_question:标签以及标签下所有问题的集合{标签：问题列表}
    :param tag:标签
    :param q:需要新增的问题
    :return:
    """
    if tag_question.get(tag) is not None:
        questions = tag_question[tag]
        questions.append(q)
        tag_question[tag] = questions
    else:
        questions = list()
        questions.append(q)
        tag_question[tag] = questions


def add_sim_question_under_tag(tag_question, tag, q):
    """
    把问题加入到标签下的问题列表中
    1、先检查tag_question是否存在标签，不存在就做插入操作
    2、如果存在，去除问题列表对象，查看q是否已经存在于列表，存在则抛出警告，不存在就插入操作
    :param tag_question:标签以及标签下所有问题的集合{标签：问题列表}
    :param tag:标签
    :param q:需要新增的问题
    :return:
    """
    if tag_question.get(tag) is not None:
        questions = tag_question[tag]
        questions.append(q)
        tag_question[tag] = questions
    else:
        log.warn('warning 相似问题的标签{}不存在！'.format(tag))


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
        writer.writerow(['content', 'label_id','label'])
        for tagKey in tag_question.keys():
            questionList = tag_question.get(tagKey)
            for question in questionList:
                writer.writerow([question, tagKey, tagKey])


with open(source_file, 'r+', encoding='utf-8') as f:
    # 载入json内容
    jsonContent = json.load(f)
    # {}部分是字典类型 []部分是列表类型
    log.info(type(jsonContent))
    log.info(type(jsonContent['data']))
    # 获得data节点的内容
    all_question = jsonContent['data']

    # {tag: [desc, answer] , ...}
    tag_desc_answer = dict()
    # {tag: [question,question] , ...}
    tag_question = dict()
    cnt_std_questions = 0
    cnt_sim_questions = 0
    for question in all_question:
        parent_id = question['parent_id']
        # 是否为标准问题
        is_std_question = 0
        if parent_id == '0':
            # parent_id=='0'没有上级问题，即为标准问题
            is_std_question = 1
            cnt_std_questions = cnt_std_questions + 1
        elif parent_id != '0':
            # parent_id<>'0'存在上级问题，即为相似问题
            is_std_question = 0
            cnt_sim_questions = cnt_sim_questions + 1

        if is_std_question == 1:
            # 如果是标准问题
            tag = question['knowledge_id']
            q = question['question']
            # ?? answer需要去标签化
            answer = question['answer']
            # 去除HTML标
            beau = BeautifulSoup(answer, "html.parser")
            answer = beau.get_text()
            # 去除空格
            answer = str(answer).replace('\n', '')

            if tag_desc_answer.get(tag) is not None:
                log.warn('重复标签：  \"knowledge_id\": \"{}\"'.format(tag))
            else:
                tag_desc_answer[tag] = [q, answer]

                add_std_question_under_tag(tag_question, tag, q)
        else:
            # 如果是相似问题
            tag = question['parent_id']
            q = question['question']
            add_sim_question_under_tag(tag_question, tag, q)

    log.info('统计数据：')
    cnt_all = 0
    for tag in tag_desc_answer.keys():
        q_a = tag_desc_answer.get(tag)
        q_list = tag_question.get(tag)
        log.info(' - 标签[{}]有问题{}个: {}  --  {}'.format(tag, len(q_list), q_a[0], q_a[1]))
        cnt_all = cnt_all + len(q_list)

    log.info("一共有{}个标签".format(tag_desc_answer.__len__()))
    log.info("一共有{}个标准问题".format(cnt_std_questions))
    log.info("一共有{}个相似问题".format(cnt_all))

    dumpTagInfo(tag_desc_answer)
    dumpTagQuestions(tag_question)
    dumpLabeldedTrainData(tag_question)
