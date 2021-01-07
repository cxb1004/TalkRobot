"""
按行读入msg.txt文件内容，
去除前后空行并以tab划分行数据
如果划分之后数据不是4个（talk_id/type/time/msg），即原始数据有问题，报警跳过，进入下行数据
如果对话类型不是g/k/l,跳过
对话文本执行如下：
如果包含了手机、电话、邮箱的标签，直接跳过
对文本进行数据清洗：去除html，去除img，去除表情
对清洗之后的文本进行相似度判断
如果没有判断结果，label_id和label为空
对话文本进行分词
把分词、label_id、label写入csv
把分词结果写入corpus.txt
统计每行的词数，用于优化特征向量的训练
"""

"""
导入系统类库并设定运行根目录
"""
import os
import sys

# 当前目录
curPath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(curPath)

"""
导入第三方类库
"""
import csv
import re
import jieba
from bs4 import BeautifulSoup

"""
导入我方项目类库
"""
from common.log import Log
from common.db import Database
from common.textSimilarity import CosSim

log = Log()

# 同级的data目录作为数据文件的根目录
dataPath = os.path.join(curPath, 'data')

# 标签库标准问题数据
label_standard_question_info = None


def is_contain_mobile_phone_email(text):
    """
    判断文本中是否包含了如下内容：[mobile][email][phone]
    :param text:
    :return:
    """
    if text.lower().find('[mobile]') > -1 or text.lower().find('[email]') > -1 or text.lower().find('[phone]') > -1:
        return True
    return False


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
    if max_similarity_value >= 0.5:
        return max_similarity_label_id, max_similarity_label, max_similarity_value
    else:
        return None, None, None


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


'''
从原始语料库文件msg.txt抽取数据，生成训练数据
1、定义输入文件msg.txt
2、定义输出文件labeledTrainData.csv 、 corpus.txt
3、运行前检查labeledTrainData.csv 、 corpus.txt是否存在，存在就删除，然后重建
4、
'''
# 读入最原始的语料库文件
MSG_FILE_NAME = 'msg.txt'
MSG_FILE = os.path.join(dataPath, MSG_FILE_NAME)

# 输出文件：打标签的训练数据
LABELEDTRAINDATA_FILE_NAME = 'labeledTrainData.csv'
labeled_train_data_file = os.path.join(dataPath, LABELEDTRAINDATA_FILE_NAME)

# 输出文件：语料库文件
CORPUS_FILE_NAME = 'corpus.txt'
corpus_file = os.path.join(dataPath, CORPUS_FILE_NAME)

# 创建打标签数据文件labeledTrainData.csv，如果存在就先删除
if os.path.isfile(labeled_train_data_file):
    os.remove(labeled_train_data_file)
    log.info('labeledTrainData.csv文件已经存在，自动清除')
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

# 统计词频用
word_count_line_dict = {}
with open(corpus_file, 'w', encoding='utf-8', newline='') as corpusFile:
    with open(labeled_train_data_file, 'w', encoding='utf-8', newline='') as labeledTrainDataFile:

        csv_writer = csv.writer(labeledTrainDataFile)
        # 在csv文件里面加入第一行title
        csv_writer.writerow(['content', 'label_id', 'label'])

        # 打开msg.txt文件流
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
                    log.warn('第{}行数据不符合规范，系统自动忽略！'.format(i))
                    continue

                # 获得对话类型
                talk_type = line_data[1]
                # type = g(访客留言) k（快捷提问） l（访客留言）的文本信息
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
                        if label_id is not None:
                            # label_id = ''
                            # label = ''
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

# ==============================整理训练数据==============================
