"""
中文短文本测试
1、使用02_chinese_corpus/data/corpus_6_4000里的数据作为基础数据
2、读取所有文件数据，循环，并作一下操作：
2.1 解析文件名，获得标签文本，根据静态字段，获得标签ID
2.2 读取文件，拼接多行文本，去掉多余空格，
2.3 对文本进行清洗（去掉HTML、去掉除问好以外的标点符号、去掉表情符等、去掉图片标签）
2.4 对文本进行初步截断，并进行分词操作，对分词结果进行二次截断，保留50个字符长度
2.5 最后形成的预料，是经过清洗以后的文本、经过分词以后，保留前50个分词结果
3、把分词之后文本、标签ID、标签文本，写入csv文件
"""
import os
import sys

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

import csv
import re
import jieba
from bs4 import BeautifulSoup
from zhon.hanzi import punctuation as punctuation
import string

from common.log import Log
from common.utils import removeFileIfExists
from common.utils import replaceMutiSpace

log = Log()

# xuyao
fileter_punctuation = (string.punctuation + punctuation) \
    .replace('?', '') \
    .replace('？', '')


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


def cleanText(text):
    """
    对文本进行清洗
    1、去除html标签
    2、去除[img]
    3、去除自定义的表情符号 {53c_min#xx#}
    4、替换部分标点符号，考虑到中文标点符号对于语义的影响，不清除标点符号
    :param text: 文本
    :return: 去除html标签
    """
    # 去掉HTML标签
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

    # 去除所有中英文标点符号：对于短文本来说，标点符号对于语义来说没有太大影响，保留了问号
    global fileter_punctuation
    new_text = re.sub("[{}]+".format(punctuation), " ", new_text)
    return new_text


# 设置语料库每一句的最多词数
word_limit_size = 50

# source_data_folder = 'D:/prj_github/TalkRobot/test/02_chinese_corpus/data/corpus_6_4000'
source_data_folder = '/python/TalkRobot/test/02_chinese_corpus/data/corpus_6_4000'

labeled_train_data_csv = os.path.join(basePath, 'data/labeledTrainData.csv')

removeFileIfExists(labeled_train_data_csv)

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

    # 获得目录下的所有文件、子目录、根目录信息
    for root, dirs, allFiles in os.walk(source_data_folder):
        cnt_files = len(allFiles)
        log.info('一共有{}个文件'.format(cnt_files))

        line_data = None
        content = None
        content_list = None
        label = None
        label_id = None

        wipFileNum = 0
        # 循环当前目录下所有文件
        for file in allFiles:
            # 打印处理进度
            log.debug('进度：{}/{}'.format(wipFileNum, cnt_files))

            # 获得文件全路径
            full_file_path = os.path.join(root, file)
            # 解析出标签，并获得标签ID
            file_name = file
            label = file.split('_')[0]
            label_id = label_id_dict[label]

            # 获得文件内容（可能多行）
            with open(full_file_path, 'r', encoding='utf-8') as f:
                content_list = f.readlines()

            # 合并同一个文件里面的多行数据，最后把同一行字符串里的多个空格替换成一个空格
            content = ''
            for content_line in content_list:
                if not (content_line == '\n' or content_line.strip() == ''):
                    # 非空行直接拼接字符串
                    content = content + content_line.replace('\n', '')
                else:
                    # 空行就拼接一个空字符
                    content = content + ' '

            # 去掉超过一个的空格
            content = replaceMutiSpace(content)

            # 清洗数据
            content = cleanText(content)

            # 先获得4倍字数限制的问题，一方面减少jieba分词的性能问题，一方面又能保留较完整的主体词语
            content = content[0:word_limit_size * 4]
            content = segment(content)
            content = ' '.join(content.split(' ')[0:word_limit_size])

            # 写入csv文件
            line_data = [content, label_id, label]
            labeled_train_data_writer.writerow(line_data)

            wipFileNum = wipFileNum + 1

log.info('训练数据整理完毕')
