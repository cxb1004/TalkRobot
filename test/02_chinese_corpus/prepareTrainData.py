"""
1、读入data/corpus_6_4000，循环目录，获得文件名和文件内容
文件名解析，解析出标签，并查询对应的标签ID,
读取文件内容，拼接多行数据，
"""
import os
import sys

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

import csv

from common.log import Log

log = Log()

source_data_folder = os.path.join(basePath, 'data/corpus_6_4000')

labeled_train_data_csv = os.path.join(basePath, 'data/labeledTrainData.csv')

if os.path.isfile(labeled_train_data_csv):
    os.remove(labeled_train_data_csv)
    log.info('labeledTrainData.csv文件已经存在，自动删除')

# 定义标签和对应的ID，用于打标签
label_id_dict = {'Auto': 0, 'Culture': 1, 'Economy': 2, 'Medicine': 3, 'Military': 4, 'Sports': 5}

# 反转标签的ID和标签值，用于查询
id_label_dict = {v: k for k, v in label_id_dict.items()}

with open(labeled_train_data_csv, 'w', encoding='utf-8', newline='') as csv_file:
    # 创建写入对象
    labeled_train_data_writer = csv.writer(csv_file)

    # 写入第一行的header
    line_data = ['content', 'label_id', 'label']
    labeled_train_data_writer.writerow(line_data)

    # 获得数据目录下的所有文件
    for root, dirs, files in os.walk(source_data_folder):
        cnt_files = len(files)
        log.info('一共有{}个文件'.format(cnt_files))

        line_data = None
        content = None
        content_list = None
        label = None
        label_id = None

        # 循环所有文件
        process = 0
        for file in files:
            full_file_path = os.path.join(root, file)
            file_name = file
            label = file.split('_')[0]
            label_id = label_id_dict[label]

            with open(full_file_path, 'r', encoding='utf-8') as f:
                content_list = f.readlines()

            content = ''
            for content_line in content_list:
                content = content + content_line

            content = content.replace("\n"," ").strip(" ")

            line_data = [content, label_id, label]
            labeled_train_data_writer.writerow(line_data)

            process = process + 1
            log.debug('读取进度：{}/{}'.format(process, cnt_files))

log.info('训练数据整理完毕')
