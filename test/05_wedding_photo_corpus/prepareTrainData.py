"""

"""
import json
import os
import sys

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

from common.log import Log

log = Log()

source_file = os.path.join(basePath, 'data/response.txt')
tag_answer = os.path.join(basePath, 'data/tag_answer.json')
tag_question = os.path.join(basePath, 'data/tag_question.json')
labeled_train_data_csv = os.path.join(basePath, 'data/labeledTrainData.csv')


def add_question_under_tag(tag_question, tag, q):
    """
    TODO
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
        questions.insert(q)
        tag_question[tag] = questions
    else:
        questions = list()
        questions.append(q)
        tag_question[tag] = questions
    pass


with open(source_file, 'r+', encoding='utf-8') as f:
    # 载入json内容
    jsonContent = json.load(f)
    # {}部分是字典类型 []部分是列表类型
    print(type(jsonContent))
    print(type(jsonContent['data']))
    # 获得data节点的内容
    all_question = jsonContent['data']

    tag_desc_answer = dict()
    tag_question = dict()
    cnt_std_questions = 0
    cnt_sim_questions = 0
    for question in all_question:
        parent_id = question['parent_id']
        # 是否为标准问题
        is_std_question = 0
        if (parent_id == '0'):
            # parent_id=='0'没有上级问题，即为标准问题
            is_std_question = 1
            cnt_std_questions = cnt_std_questions + 1
        else:
            # parent_id<>'0'存在上级问题，即为相似问题
            is_std_question = 0
            cnt_sim_questions = cnt_sim_questions + 1

        if is_std_question == 1:
            # 如果是标准问题
            tag = question['knowledge_id']
            q = question['question']
            # ?? answer需要去标签化
            answer = question['answer']

            if tag_desc_answer.get(tag) is not None:
                log.warn('重复标签：  \"knowledge_id\": \"{}\"'.format(tag))
            else:
                tag_desc_answer[tag] = [q, answer]

                add_question_under_tag(tag_question, tag, q)

        else:
            pass

        # 使用knowledge_id作为tag

        # 使用标准问题作为这个标签的描述
        tag_desc = question['question']

        print(tag)
