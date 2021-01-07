"""
【概要功能】
    本程序主要是把婚纱摄影的语料库转化为训练文件
【输入输出】：
    - 读入
        53data/wedding_photo/msg.txt     <talk_id><tab><type><tab><msg_time><tab><msg>
    - 输出文件
        53data/wedding_photo/yyyymmdd/labeledTrainData.csv  <文本> <分类编码Code>
        53data/wedding_photo/yyyymmdd/corpus.txt
【主要流程】
    1、读取msg.txt文件，按行读取文件内容（跳过第一行）,以tab区分各个字段
    2、读取type = g(访客留言) k（快捷提问） l（访客留言）的文本信息
       o（退出挽留） q（快问快答） z（访客表单） 这几个属于访客留下的联系信息，不属于提问，因而忽略
    3、数据清洗：去除Html，获得最终的文本
    4、从数据库中，获取每个标签的标准句  label_id, sentences(多个)
    5、把文本和每个标签的每个标准语句进行相似度比较，对每个label取平均值
    6、根据每个label的相似度，取最高的相似度，并且高于阀值
    7、把清洗之后的文本和label_id，以csv的格式保存

"""
import sys
import warnings
import os
import time
from bs4 import BeautifulSoup
import string
from zhon.hanzi import punctuation
import jieba
import csv
import re



# 自定义的导入目录
from common.config import Config as ProjectConfig
from common.log import ProjectLog
from common.db import Database
from textSimilarity import CosSim

warnings.filterwarnings("ignore")
log = ProjectLog()
config = ProjectConfig()
PROJECT_ROOT = config.get_project_root_dir()
DATA_ROOT = os.path.join(PROJECT_ROOT, '53data')
# 日期，用于工作目录生成
cur_date = time.strftime('%Y%m%d', time.localtime(time.time()))
industry_name = 'wedding_photo'

punctuation_en = string.punctuation
punctuation_cn = punctuation
punctuation_str = punctuation_en + punctuation_cn

# 通用库
LABEL_COMMON_LIB = '0001'
# 行业库：婚纱摄影
LABEL_INDUSTRY_LIB_WEDDING_PHOTO = '0002'
# 标签库标准问题数据
label_standard_question_info = None

#
question_rank_dict = {}


def clear_content(text):
    """
    对文本进行清洗
    1、去除html标签
    2、去除[img]
    3、去除自定义的表情符号 {53c_min#xx#}

    :param text: 文本
    :return: 去除html标签
    """
    beau = BeautifulSoup(text)
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


def is_contain_mobile_phone_email(text):
    """
    判断文本中是否包含了如下内容：[mobile][email][phone]
    :param text:
    :return:
    """
    if text.lower().find('[mobile]') > -1 or text.lower().find('[email]') > -1 or text.lower().find('[phone]') > -1:
        return True
    return False


def get_max_similarity_label_id(text):
    """
    用文本相似度算法，比较文本和各个标签下标准语句的相似度
        1、对每个标签下的一组值进行平均计算，选择最高值（平均值有可能让最相似度的值失去了作用）
        2、获取最高的那个标准句对应的标签 （选择这种方案）
    总体流程：
    1、获取行业库标签 / 通用库标签的ID、标签名、标准句子
    :param text:预测文本
    :return: 最有可能的label_id
    """
    global label_standard_question_info
    if label_standard_question_info is None:
        log.info('获取通用库、行业库（婚纱摄影）的标签标准问题信息:')
        db = Database()
        sql = 'SELECT q.label_id, l.label, q.question, l.industry_code, q.id qid FROM robot_label l, robot_label_std_q q WHERE l.industry_code IN (0001 , 0002) AND l.id = q.label_id ORDER BY l.industry_code ASC, l.id ASC, q.id ASC;'
        try:
            label_standard_question_info = db.query(sql, None)
            log.info('获取成功！')
        except Exception as e:
            log.error('数据库操作失败：{}'.format(sql))
            log.error(e)
            exit(999)
        finally:
            db.close()

    log.debug('开始计算相似度：')
    sim = CosSim()

    log.debug('主文本：{}'.format(text))
    max_similarity_value = None
    max_similarity_label_id = None
    max_similarity_label = None
    for question_info in label_standard_question_info:
        cur_question = str(question_info[2])
        cur_label_id = int(question_info[0])
        cur_label = str(question_info[1])
        if (not (cur_question is None)) and len(cur_question.strip()) > 0:
            cur_similarity_value = sim.getSimilarityIndex(text, cur_question)

            if max_similarity_value is None:
                max_similarity_value = cur_similarity_value
                max_similarity_label = str(question_info[1])
                max_similarity_label_id = float(question_info[0])
                log.debug(
                    '   胜出：相似度[{}] 标准文本[{}] 所属标签[{} - {}]'.format(cur_similarity_value, cur_question, cur_label_id,
                                                                  cur_label))
            elif cur_similarity_value >= max_similarity_value:
                max_similarity_value = cur_similarity_value
                max_similarity_label = str(question_info[1])
                max_similarity_label_id = int(question_info[0])
                log.debug(
                    '   胜出：相似度[{}] 标准文本[{}] 所属标签[{} - {}]'.format(cur_similarity_value, cur_question, cur_label_id,
                                                                  cur_label))
            else:
                log.debug(
                    '   落选：相似度[{}] 标准文本[{}] 所属标签[{} - {}]'.format(cur_similarity_value, cur_question, cur_label_id,
                                                                  cur_label))
    # TODO 优化内容
    # 1、这里的阀值可以写入数据库
    # 2、这里可以增加一个命中率热度的指标，每个最终胜出的标准问题，热度+1。 有助于对这些问题的管理，比如热度第的问句可以删除或修改
    if max_similarity_value >= 0.3:
        return max_similarity_label_id, max_similarity_label, max_similarity_value
    else:
        return None, None, None


# 读入文件
MSG_FILE_NAME = 'msg.txt'
MSG_FILE = os.path.join(DATA_ROOT, industry_name)
MSG_FILE = os.path.join(MSG_FILE, MSG_FILE_NAME)

DATA_INDUSTRY_PATH = os.path.join(DATA_ROOT, industry_name)
DATA_INDUSTRY_DATE_PATH = os.path.join(DATA_INDUSTRY_PATH, cur_date)
if not os.path.exists(DATA_INDUSTRY_DATE_PATH):
    # 创建目录
    os.makedirs(DATA_INDUSTRY_DATE_PATH)
    log.info('创建模型训练数据基础目录：' + DATA_INDUSTRY_PATH)

# 输出文件：打标签的训练数据
LABELEDTRAINDATA_FILE_NAME = 'labeledTrainData.csv'
labeled_train_data_file = os.path.join(DATA_INDUSTRY_DATE_PATH, LABELEDTRAINDATA_FILE_NAME)
# 输出文件：语料库文件
CORPUS_FILE_NAME = 'corpus.txt'
corpus_file = os.path.join(DATA_INDUSTRY_DATE_PATH, CORPUS_FILE_NAME)

# 创建打标签数据文件labeledTrainData.csv，如果存在就先删除
if os.path.isfile(labeled_train_data_file):
    os.remove(labeled_train_data_file)
    log.info('labeledTrainData.csv文件已经存在，自动删除')
try:
    # 在linux上有效
    os.mknod(labeled_train_data_file)
except Exception as e:
    # 对win操作系统的处理
    fp = os.open(labeled_train_data_file, os.O_CREAT)
    os.close(fp)
log.info('创建labeledTrainData.csv文件：{}'.format(labeled_train_data_file))

# 创建语料库文件corpus.txt，如果存在就先删除
if os.path.isfile(corpus_file):
    os.remove(corpus_file)
    log.info('corpus_file文件已经存在，自动删除')
try:
    # 在linux上有效
    os.mknod(corpus_file)
except Exception as e:
    # 对win操作系统的处理
    fp = os.open(corpus_file, os.O_CREAT)
    os.close(fp)
log.info('创建corpus_file文件：' + corpus_file)

word_count_line_dict = {}
with open(corpus_file, 'w', encoding='utf-8', newline='') as corpusFile:
    with open(labeled_train_data_file, 'w', encoding='utf-8', newline='') as labeledTrainDataFile:
        # 在csv文件里面加入第一行title
        csv_writer = csv.writer(labeledTrainDataFile)
        csv_writer.writerow(['content', 'label_id', 'label'])

        with open(MSG_FILE, 'r', encoding='utf-8') as msgFile:
            lines = msgFile.readlines()
            total = len(lines)
            i = 1
            for line in lines:
                log.info('进度：{}/{}'.format(i, total))
                i = i + 1
                line_data = line.strip('\n').split('\t')

                # 判断行记录的格式是否正确
                if not len(line_data) == 4:
                    # 如果数据格式不对，直接跳过该条数据
                    continue

                talk_type = line_data[1]
                if talk_type == 'g' or talk_type == 'k' or talk_type == 'l':
                    text = line_data[3]

                    if is_contain_mobile_phone_email(text):
                        # 如果包含了手机、电话、邮箱，文本就不是访客问话，跳过
                        continue

                    # 去除html标签
                    text = clear_content(text)

                    # 判断是否为空字符串
                    if len(text.strip()) > 0:
                        maxSimilarity = get_max_similarity_label_id(text)
                        label_id = maxSimilarity[0]
                        label = maxSimilarity[1]
                        if label_id is None:
                            label_id = ''
                            label = ''
                        text = segment(text)

                        csv_writer.writerow([text, label_id, label])
                        corpusFile.write(text + '\n')

                        # 统计一行的字数
                        wordCound = len(text.replace(' ', ''))
                        if wordCound in word_count_line_dict:
                            cnt = word_count_line_dict[wordCound]
                            cnt = cnt + 1
                            word_count_line_dict[wordCound] = cnt
                        else:
                            word_count_line_dict[wordCound] = 1

log.info('labeledTrainData.csv文件写入完毕：{}'.format(labeled_train_data_file))
log.info('corpus.txt文件写入完毕：{}'.format(corpus_file))
log.info('行字数统计：{}'.format(word_count_line_dict))
