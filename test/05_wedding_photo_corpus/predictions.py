"""
1、读入label2idx.json / word2idx.json
2、载入模型文件和参数
3、开启输入接口，显示操作菜单
4、输入成功之后，进行预测，完成之后返回菜单
"""
"""
导入系统类库并设定运行根目录
"""
import csv
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


# 定义标签和对应的ID，用于打标签
# 标签和数字id的对应转换，另一块代码在_genVocabulary中，重构的时候选择一个方案，然后两个地方选择一处修改
def getTagDictory():
    tagFile = os.path.join(basePath, 'data/tag_info.csv')
    labelinfo = {}
    # idx = 0
    with open(tagFile, 'r', encoding='utf-8') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if reader.line_num != 1:
                labelinfo.update({row[0]: row[0]})
                # idx = idx + 1
    log.info('读入标签{}个'.format(labelinfo.__len__()))
    return labelinfo


label_id_dict = getTagDictory()
id_label_dict = {v: k for k, v in label_id_dict.items()}

sequenceLength = 13
# 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
file_word2idx_json = os.path.join(basePath, 'data/word2idx.json')
with open(file_word2idx_json, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

file_label2idx_json = os.path.join(basePath, 'data/label2idx.json')
with open(file_label2idx_json, "r", encoding="utf-8") as f:
    label2idx = json.load(f)
idx2label = {value: key for key, value in label2idx.items()}

file_tag_info_csv = os.path.join(basePath, 'data/tag_info.csv')
tag2idx = {}
with open(file_tag_info_csv, "r", encoding="utf-8") as f:
    tagInfo = csv.reader(f)
    for tagData in tagInfo:
        if tagInfo.line_num != 1:
            tag2idx[tagData[0]] = tagData[1]

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

        # ===============开始读取原始语料库，进行数据验证==================================
        # 1、读取原始语料库（最好是没有经过清洗、分词的）
        # 2、获得标签，标准问题是knowledge_id / 相似问题 parent_id
        source_data_folder = os.path.join(basePath, 'data/response.txt')
        result_file = os.path.join(basePath, 'data/result.csv')

        with open(result_file, 'w', encoding='utf-8') as result:
            writer = csv.writer(result)
            writer.writerow(['content', 'expected', 'actual', 'isPass'])

            with open(source_data_folder, 'r+', encoding='utf-8') as f:
                jsonContent = json.load(f)
                all_data = jsonContent['data']
                for data in all_data:
                    knowledge_id = data['knowledge_id']
                    x = data['question']
                    parent_id = data['parent_id']

                    if parent_id == '0':
                        expect_tag = knowledge_id
                    else:
                        expect_tag = parent_id

                    cut_list = jieba.lcut(x)
                    # 从词库里面匹配对应的词，如果没有，就用‘UNK’替代
                    xIds = [word2idx.get(item, word2idx["UNK"]) for item in cut_list]
                    # log.debug('源文本词数：{}'.format(len(cut_list)))
                    # log.debug('特征词词数：{}'.format(len(xIds)))
                    if len(xIds) >= sequenceLength:
                        xIds = xIds[:sequenceLength]
                    else:
                        xIds = xIds + [word2idx["PAD"]] * (sequenceLength - len(xIds))
                    # log.debug('截断特征词词数：{}'.format(len(xIds)))
                    if len(xIds) >= sequenceLength:
                        xIds = xIds[:sequenceLength]
                    else:
                        xIds = xIds + [word2idx["PAD"]] * (sequenceLength - len(xIds))

                    pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]
                    actual = idx2label.get(pred)
                    log.debug('\n 正确标签：{} \n 预测标签：{} \n 读入文本：{} \n 预测问题：{} '.format(expect_tag, actual, x, tag2idx[actual]))

                    if expect_tag == actual:
                        is_pass = 1
                    else:
                        is_pass = 0

                    writer.writerow([x, expect_tag, actual, is_pass])

# pred = [idx2label[item] for item in pred]
# print("最终预测结果为{}".format(idx2label[pred]))
log.info('运行完毕')
