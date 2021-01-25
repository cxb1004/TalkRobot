import json
import os
import sys

import jieba
import tensorflow.compat.v1 as tf

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

from common.log import Log

log = Log()


class Prediction(object):
    word2idx = None
    label2idx = None
    graph = None
    sess = None
    inputX = None
    dropoutKeepProb = None
    predictions = None
    label_id_dict = None
    id_label_dict = None

    sequenceLength = 200

    def __init__(self):
        file_word2idx_json = os.path.join(basePath, 'data/word2idx.json')
        with open(file_word2idx_json, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)

        file_label2idx_json = os.path.join(basePath, 'data/label2idx.json')
        with open(file_label2idx_json, "r", encoding="utf-8") as f:
            self.label2idx = json.load(f)

        # 定义标签和对应的ID，用于打标签
        self.label_id_dict = {'Auto': 0, 'Culture': 1, 'Economy': 2, 'Medicine': 3, 'Military': 4, 'Sports': 5}

        # 反转标签的ID和标签值，用于查询
        self.id_label_dict = {v: k for k, v in self.label_id_dict.items()}
        log.info('载入标签库完成！')

        self.graph = tf.Graph()
        with self.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                          gpu_options=gpu_options)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                modelPath = os.path.join(basePath, 'data/model/')
                # 第二种checkpoint模型导入方式，因为有乱码，无法载入
                log.info('准备载入模型文件：' + modelPath)
                checkpoint_file = tf.train.latest_checkpoint(modelPath)
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                log.info('载入模型完毕')

                # 获得需要喂给模型的参数，输出的结果依赖的输入值
                self.inputX = self.graph.get_operation_by_name("inputX").outputs[0]
                self.dropoutKeepProb = self.graph.get_operation_by_name("dropoutKeepProb").outputs[0]

                # 获得输出的结果 (0不能改成1，否则会有异常KeyError: "The name 'output/predictions:1' refers to a Tensor which does not exist. The operation, 'output/predictions', exists but only has 1 outputs.")
                self.predictions = self.graph.get_tensor_by_name("output/predictions:0")

    def pred(self, str):

        # cxb变更：需要用jieba进行中文分词
        cut_list = jieba.lcut(str)
        xIds = [self.word2idx.get(item, self.word2idx["UNK"]) for item in cut_list]
        if len(xIds) >= self.sequenceLength:
            xIds = xIds[:self.sequenceLength]
        else:
            xIds = xIds + [self.word2idx["PAD"]] * (self.sequenceLength - len(xIds))

        pred = self.sess.run(self.predictions, feed_dict={self.inputX: [xIds], self.dropoutKeepProb: 1.0})[0]
        log.debug(pred)
        log.debug(self.id_label_dict[pred])

        return self.id_label_dict[pred]


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
