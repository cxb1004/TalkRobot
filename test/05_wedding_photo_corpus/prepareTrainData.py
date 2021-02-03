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

with open(source_file, 'r+', encoding='utf-8') as f:
    # 载入json内容
    jsonContent = json.load(f)
    # {}部分是字典类型 []部分是列表类型
    print(type(jsonContent))
    print(type(jsonContent['data']))
    # 获得data节点的内容
    all_question = jsonContent['data']

    dict_tag = {}
    tag_question = []
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
            desc = question['question']
            answer = question['answer']

            if dict_tag.has_key(tag):
                log.warn('重复标签：  \"knowledge_id\": \"{}\"'.format(tag))
            else:
                dict_tag[tag] = desc
        else:


        # 使用knowledge_id作为tag

        # 使用标准问题作为这个标签的描述
        tag_desc = question['question']

        print(tag)
