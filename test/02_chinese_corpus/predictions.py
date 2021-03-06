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
import jieba
import string
from zhon.hanzi import punctuation
from bs4 import BeautifulSoup

# 全局变量
# 定义标签和对应的ID，用于打标签
LABEL_ID_DICT = {'Auto': 0, 'Culture': 1, 'Economy': 2, 'Medicine': 3, 'Military': 4, 'Sports': 5}
# 反转标签的ID和标签值，用于查询
ID_LABEL_DICT = {v: k for k, v in LABEL_ID_DICT.items()}

punctuation_en = string.punctuation
punctuation_cn = punctuation
punctuation_str = punctuation_en + punctuation_cn


def clearContent(str):
    # log.debug('去除html标签前：' + lingString)
    beau = BeautifulSoup(str, features="lxml")
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


x = "外地乙肝病毒携带者拿到从事食品行业健康证的先例，让在北京的小廖感到欣慰，但同是乙肝病毒携带者的他前往相关机构申请办理该证却碰壁。11月16日上午，小廖一纸诉状递到东城法院"
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

x = clearContent(x)
xIds = [word2idx.get(item, word2idx["UNK"]) for item in jieba.lcut(x)]
if len(xIds) >= sequenceLength:
    xIds = xIds[:sequenceLength]
else:
    xIds = xIds + [word2idx["PAD"]] * (sequenceLength - len(xIds))

graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        print("载入CheckPoint模型文件目录：{}".format(modelPath))
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
        pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]
        print('pred is {}'.format(pred))

print('输入文本为：{}'.format(x))
print("最终预测结果为{}".format(idx2label[pred]))
print(ID_LABEL_DICT)
