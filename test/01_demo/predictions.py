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

from common.log import Log

log = Log()

# x = "it must be assumed that those who praised this film the greatest filmed opera ever didnt i read somewhere either dont care for opera dont care for wagner or dont care about anything except their desire to appear cultured either as a representation of wagners swan-song or as a movie this strikes me as an unmitigated disaster with a leaden reading of the score matched to a tricksy lugubrious realisation of the textits questionable that people with ideas as to what an opera or for that matter a play especially one by shakespeare is about should be allowed anywhere near a theatre or film studio; syberberg very fashionably but without the smallest justification from wagners text decided that parsifal is about bisexual integration so that the title character in the latter stages transmutes into a kind of beatnik babe though one who continues to sing high tenor -- few if any of the actors in the film are the singers and we get a double dose of armin jordan the conductor who is seen as the face but not heard as the voice of amfortas and also appears monstrously in double exposure as a kind of batonzilla or conductor who ate monsalvat during the playing of the good friday music -- in which by the way the transcendant loveliness of nature is represented by a scattering of shopworn and flaccid crocuses stuck in ill-laid turf an expedient which baffles me in the theatre we sometimes have to piece out such imperfections with our thoughts but i cant think why syberberg couldnt splice in for parsifal and gurnemanz mountain pasture as lush as was provided for julie andrews in sound of musicthe sound is hard to endure the high voices and the trumpets in particular possessing an aural glare that adds another sort of fatigue to our impatience with the uninspired conducting and paralytic unfolding of the ritual someone in another review mentioned the 1951 bayreuth recording and knappertsbusch though his tempi are often very slow had what jordan altogether lacks a sense of pulse a feeling for the ebb and flow of the music -- and after half a century the orchestral sound in that set in modern pressings is still superior to this film"
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

        csv_file = os.path.join(basePath, 'data/preProcess/labeledTrain.csv')
        result_file = os.path.join(basePath, 'data/result.csv')

        with open(csv_file, 'r', encoding='utf-8') as source:
            with open(result_file, 'w', encoding='utf-8') as result:
                alllines = csv.reader(source)
                writer = csv.writer(result)

                writer.writerow(['content', 'expected', 'actual', 'isPass'])

                for line in alllines:
                    log.info('进度：{}/{}'.format(alllines.line_num, '25001'))
                    if not alllines.line_num == 1:
                        x = line[0]
                        expected = line[2]

                        xIds = [word2idx.get(item, word2idx["UNK"]) for item in x.split(" ")]
                        if len(xIds) >= sequenceLength:
                            xIds = xIds[:sequenceLength]
                        else:
                            xIds = xIds + [word2idx["PAD"]] * (sequenceLength - len(xIds))

                        pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0})[0]
                        actual = pred

                        if str(actual) == str(expected):
                            isPass = 1
                        else:
                            isPass = 0

                        writer.writerow([x, expected, actual, isPass])

# pred = [idx2label[item] for item in pred]
# print("最终预测结果为{}".format(idx2label[pred]))
log.info('运行完毕')
