"""
1、读入label2idx.json / word2idx.json
2、载入模型文件和参数
3、开启输入接口，显示操作菜单
4、输入成功之后，进行预测，完成之后返回菜单
"""

import sys
import warnings
import os
import json
import tensorflow.compat.v1 as tf
import string
import jieba
from zhon.hanzi import punctuation
from bs4 import BeautifulSoup

from common1.log import ProjectLog
from common1.config import Config as ProjConfig

warnings.filterwarnings("ignore")
log = ProjectLog()
projConfig = ProjConfig()
ROOT_PATH = projConfig.get_project_root_dir()

MODEL_PATH = os.path.join(ROOT_PATH, '53model')
ACTIVE_MODEL_PATH = os.path.join(ROOT_PATH, '53model/Bi-LSTM-atten/model')
SEQUENCE_LENGTH = 200

punctuation_en = string.punctuation
punctuation_cn = punctuation
punctuation_str = punctuation_en + punctuation_cn
log.debug('准备过滤的标点符号：' + punctuation_str)

prediction = None


class Prediction(object):
    word2idx = None
    idx2label = None
    label2idx = None
    graph = None
    sess = None
    inputX = None
    dropoutKeepProb = None
    predictions = None
    id_label_dict = None

    label_id_dict = {'Auto': 0, 'Culture': 1, 'Economy': 2, 'Medicine': 3, 'Military': 4, 'Sports': 5}

    def __init__(self):
        # 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
        with open(os.path.join(MODEL_PATH, "word2idx.json"), "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        log.info('载入词库完成')

        # with open("../data/wordJson/label2idx.json", "r", encoding="utf-8") as f:
        with open(os.path.join(MODEL_PATH, "label2idx.json"), "r", encoding="utf-8") as f:
            self.label2idx = json.load(f)
        # 把label的索引逆反一下，key变value，value变key
        self.idx2label = {value: key for key, value in self.label2idx.items()}
        log.info('载入标签库完成！')

        self.graph = tf.Graph()
        with self.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                          gpu_options=gpu_options)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # 第二种checkpoint模型导入方式，因为有乱码，无法载入
                log.info('准备载入模型文件：' + ACTIVE_MODEL_PATH)
                checkpoint_file = tf.train.latest_checkpoint(ACTIVE_MODEL_PATH)
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                log.info('载入模型完毕')

                # 获得需要喂给模型的参数，输出的结果依赖的输入值
                self.inputX = self.graph.get_operation_by_name("inputX").outputs[0]
                self.dropoutKeepProb = self.graph.get_operation_by_name("dropoutKeepProb").outputs[0]

                # 获得输出的结果 (0不能改成1，否则会有异常KeyError: "The name 'output/predictions:1' refers to a Tensor which does not exist. The operation, 'output/predictions', exists but only has 1 outputs.")
                self.predictions = self.graph.get_tensor_by_name("output/predictions:0")

        self.id_label_dict = {value: key for key, value in self.label_id_dict.items()}
        log.info('ID转标签数据载入完毕：' + str(self.id_label_dict))

    def pred(self, str):

        # cxb变更：需要用jieba进行中文分词
        x = self.clearContent(str)
        xIds = [self.word2idx.get(item, self.word2idx["UNK"]) for item in jieba.lcut(x)]
        if len(xIds) >= SEQUENCE_LENGTH:
            xIds = xIds[:SEQUENCE_LENGTH]
        else:
            xIds = xIds + [self.word2idx["PAD"]] * (SEQUENCE_LENGTH - len(xIds))

        pred = self.sess.run(self.predictions, feed_dict={self.inputX: [xIds], self.dropoutKeepProb: 1.0})[0]
        log.debug(pred)
        log.debug(self.id_label_dict[pred])

        return self.id_label_dict[pred]

    def clearContent(self, str):
        # log.debug('去除html标签前：' + lingString)
        beau = BeautifulSoup(str)
        # log.debug('去除html标签后：' + lingString)
        # 去除HTML标
        newSubject = beau.get_text()
        # log.debug('去除标点符号前：' + newSubject)
        global punctuation_en
        for i in punctuation_str:
            newSubject = newSubject.replace(i, '')
        # log.debug('去除标点符号后：' + newSubject)
        # log.debug('分词前：' + str(newSubject))
        list = jieba.cut(newSubject)
        newSubject = [word.lower() for word in list if word != ' ']
        # log.debug('分词后：' + str(newSubject))
        # # 这里相当于把分词之后的结果，用空格重新串联起来，这样后面处理只需要用空格进行分割即可
        # 待优化：已经测试出对于输入数据，是否用空格连接，对结果是有影响的。那么在训练的时候也会有影响，这个作为优化项之一，后期研究
        newSubject = "".join(newSubject)
        return newSubject


def showMenu():
    print('\n\n功能菜单：')
    op = input('请输入你要预测的文本，输入0结束程序：')

    if op == '0':
        print('\n程序关闭')
        exit(0)
    else:
        global prediction
        result = prediction.pred(op)
        print(result)
    showMenu()


prediction = Prediction()
showMenu()

# x = "今天，我们在这里举行座谈会，纪念中国共产党的优秀党员，忠诚的共产主义战士，久经考验的无产阶级革命家、杰出的军事家，我军装甲兵的主要创建者许光达同志诞辰100周年。许光达同志在长期的革命生涯中，为民族独立和人民解放，为国防和军队建设，贡献了毕生精力，建立了不可磨灭的功勋。"
#
# result = prediction.pred(x)
# print('result = ' + str(result))
# label_dict = {0: 'Auto', 1: 'Culture', 2: 'Economy', 3: 'Medicine', 4: 'Military', 5: 'Sports'}
# print(label_dict[result])

# showMenu()


# punctuation_en = string.punctuation
# punctuation_cn = punctuation
# punctuation_str = punctuation_en + punctuation_cn
# log.debug('准备过滤的标点符号：' + punctuation_str)
#
#
# def clearContent(lingString):
#     """
#     去除Html标签，并且把大写字母转化为小写
#     这部分的功能不涉及分词，并且大小写转化也对中文没有影响，因此这里不需要做任何针对中文的特殊处理
#     BeautifulSoup是python用来处理文本内容的类库，常用于数据爬取提取功能
#     :param lingString: 文本review
#     :return: 去除html标签，并转小写的文字
#     """
#     # log.debug('去除html标签前：' + lingString)
#     beau = BeautifulSoup(lingString)
#     # log.debug('去除html标签后：' + lingString)
#     # 去除HTML标
#     newSubject = beau.get_text()
#     # log.debug('去除标点符号前：' + newSubject)
#     global punctuation_en
#     for i in punctuation_str:
#         newSubject = newSubject.replace(i, '')
#     # log.debug('去除标点符号后：' + newSubject)
#     # log.debug('分词前：' + str(newSubject))
#     list = jieba.cut(newSubject)
#     newSubject = [word.lower() for word in list if word != ' ']
#     # log.debug('分词后：' + str(newSubject))
#     # # 这里相当于把分词之后的结果，用空格重新串联起来，这样后面处理只需要用空格进行分割即可
#     # 待优化：已经测试出对于输入数据，是否用空格连接，对结果是有影响的。那么在训练的时候也会有影响，这个作为优化项之一，后期研究
#     newSubject = "".join(newSubject)
#     return newSubject
#
#
# # 预测代码
# x = "今天，我们在这里举行座谈会，纪念中国共产党的优秀党员，忠诚的共产主义战士，久经考验的无产阶级革命家、杰出的军事家，我军装甲兵的主要创建者许光达同志诞辰100周年。许光达同志在长期的革命生涯中，为民族独立和人民解放，为国防和军队建设，贡献了毕生精力，建立了不可磨灭的功勋。"
# log.info('分析文本为：' + x)
#
# # 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
# with open(os.path.join(MODEL_PATH, "word2idx.json"), "r", encoding="utf-8") as f:
#     word2idx = json.load(f)
# log.info('载入词库完成')
#
# # with open("../data/wordJson/label2idx.json", "r", encoding="utf-8") as f:
# with open(os.path.join(MODEL_PATH, "label2idx.json"), "r", encoding="utf-8") as f:
#     label2idx = json.load(f)
# # 把label的索引逆反一下，key变value，value变key
# idx2label = {value: key for key, value in label2idx.items()}
# log.info('载入标签库完成！')
#
# # cxb变更：需要用jieba进行中文分词
# # xIds = [word2idx.get(item, word2idx["UNK"]) for item in x.split(" ")]
# x = clearContent(x)
# xIds = [word2idx.get(item, word2idx["UNK"]) for item in jieba.lcut(x)]
# if len(xIds) >= SEQUENCE_LENGTH:
#     xIds = xIds[:SEQUENCE_LENGTH]
# else:
#     xIds = xIds + [word2idx["PAD"]] * (SEQUENCE_LENGTH - len(xIds))
#
# graph = tf.Graph()
# with graph.as_default():
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#     session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
#     sess = tf.Session(config=session_conf)
#     with sess.as_default():
#         # 第二种checkpoint模型导入方式，因为有乱码，无法载入
#         log.info('准备载入模型文件：' + ACTIVE_MODEL_PATH)
#         checkpoint_file = tf.train.latest_checkpoint(ACTIVE_MODEL_PATH)
#         saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#         saver.restore(sess, checkpoint_file)
#         log.info('载入模型完毕')
#
#         # 获得需要喂给模型的参数，输出的结果依赖的输入值
#         inputX = graph.get_operation_by_name("inputX").outputs[0]
#         dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]
#
#         # 获得输出的结果 (0不能改成1，否则会有异常KeyError: "The name 'output/predictions:1' refers to a Tensor which does not exist. The operation, 'output/predictions', exists but only has 1 outputs.")
#         predictions = graph.get_tensor_by_name("output/predictions:0")
#
#         pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]
#
# pred = [idx2label[item] for item in [pred]]
# print(pred)
# print(pred[0])
#
# label_dict = {0: 'Auto', 1: 'Culture', 2: 'Economy', 3: 'Medicine', 4: 'Military', 5: 'Sports'}
# print(label_dict[int(pred[0])])
