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


x = "would anyone really watch this rubbish if it didnt contain little children running around nude from a cinematic point of view it is probably one of the worst films i have encountered absolutely dire some "
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

xIds = [word2idx.get(item, word2idx["UNK"]) for item in x.split(" ")]
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

# pred = [idx2label[item] for item in pred]
print("最终预测结果为{}".format(idx2label[pred]))


