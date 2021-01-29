"""
1、读入label2idx.json / word2idx.json
2、载入模型文件和参数
3、开启输入接口，显示操作菜单
4、输入成功之后，进行预测，完成之后返回菜单
"""
"""
导入系统类库并设定运行根目录
"""
import os
import sys

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

import tensorflow.compat.v1 as tf
import json
import csv
import jieba

from common.log import Log
from common.utils import replaceMutiSpace

log = Log()

# 定义标签和对应的ID，用于打标签
# label_id_dict = {'Auto': 'ID_Auto',
#                  'Culture': 'ID_Culture',
#                  'Economy': 'ID_Economy',
#                  'Medicine': 'ID_Medicine',
#                  'Military': 'ID_Military',
#                  'Sports': 'ID_Sports'}
label_id_dict = {'Auto': 'ID_Auto',
                 'Culture': 'ID_Culture'}
id_label_dict = {v: k for k, v in label_id_dict.items()}

sequenceLength = 200
# 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
file_word2idx_json = os.path.join(basePath, 'data/word2idx.json')
with open(file_word2idx_json, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

file_label2idx_json = os.path.join(basePath, 'data/label2idx.json')
with open(file_label2idx_json, "r", encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label = {value: key for key, value in label2idx.items()}

modelPath = os.path.join(basePath, 'data/model/')

graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        log.info("载入CheckPoint模型文件目录：{}".format(modelPath))
        checkpoint_file = tf.train.latest_checkpoint(modelPath)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 获得需要喂给模型的参数，输出的结果依赖的输入值
        inputX = graph.get_operation_by_name("inputX").outputs[0]
        dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

        # 获得输出的结果
        predictions = graph.get_tensor_by_name("output/predictions:0")
        # print("predictions对象：{}".format(predictions))
        # print("预测结果：{}".format(sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})))
        log.info('模型载入成功, yeah')

        source_data_folder = '/python/TalkRobot/test/02_chinese_corpus/data/corpus_6_4000'
        result_file = os.path.join(basePath, 'data/result.csv')

        with open(result_file, 'w', encoding='utf-8') as result:
            writer = csv.writer(result)
            writer.writerow(['content', 'expected', 'actual', 'isPass'])

            for root, dirs, allFiles in os.walk(source_data_folder):
                all_count = 0
                pass_count = 0
                for file in allFiles:
                    # 获得文件全路径
                    full_file_path = os.path.join(root, file)
                    # 解析出标签，并获得标签ID
                    file_name = file
                    label = file.split('_')[0]
                    expected = label_id_dict[label]

                    if expected is not None:

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

                        x = content
                        cut_list = jieba.lcut(x)
                        xIds = [word2idx.get(item, word2idx["UNK"]) for item in cut_list]
                        log.debug('源文本词数：{}'.format(len(cut_list)))
                        log.debug('特征词词数：{}'.format(len(xIds)))
                        if len(xIds) >= sequenceLength:
                            xIds = xIds[:sequenceLength]
                        else:
                            xIds = xIds + [word2idx["PAD"]] * (sequenceLength - len(xIds))
                        log.debug('截断特征词词数：{}'.format(len(xIds)))

                        pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]
                        actual = idx2label.get(pred)
                        if actual is None:
                            actual = 'None'

                        log.debug('pred={}'.format(actual))
                        log.debug('expected={}'.format(expected))
                        all_count = all_count + 1
                        if str(actual) == str(expected):
                            isPass = 1
                            pass_count = pass_count +1
                        else:
                            isPass = 0

                        log.info('成功数：{} 总数：{} 准确率:{}'.format(pass_count, all_count, format(float(pass_count) / float(all_count), '.4f')))

                        writer.writerow([x, expected, actual, isPass])

# pred = [idx2label[item] for item in pred]
# print("最终预测结果为{}".format(idx2label[pred]))
log.info('运行完毕')
