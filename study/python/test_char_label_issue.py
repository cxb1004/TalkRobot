import numpy as np
"""
【问题描述】
目前在打标签的时候，标签文本sports,标签ID 5
其中标签ID是为了适配原有代码，特意设置成数字
影响代码如下：
def _genVocabulary(self, reviews, labels):
        # set搭建无序不重复元素集，即去重
        uniqueLabel = list(set(labels))
        # 把标签数据转化为字典类型
        # 本来这里是把标签数据转为ID（以数量为上限）
        # label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        label2idx = dict(zip(uniqueLabel, uniqueLabel))
        self.labelList = list(range(len(uniqueLabel)))

def _genTrainEvalData(self, reviewIds, labelIds, word2idx, rate):
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labelIds[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labelIds[trainIndex:], dtype="float32")

那么在现实场景中，labelID很有可能是字符串，如 tag_001 / tag_webbing_photo等
在np.array(labelIds[:trainIndex], dtype="float32")就会出错

改动的影响：
原本的方案较为直观，

"""
# lableIDs = ['3', '1', '2', '3', '1', '1', '1']
lableIDs = ['a', 'c', 'b', 'c', 'a', 'a', 'a']
print('trainLabels:{}'.format(lableIDs))
# set搭建无序不重复元素集，即去重
uniqueLabel = list(set(lableIDs))
print('uniqueLabels:{}'.format(uniqueLabel))

# 把标签数据转化为字典类型
# 本来这里是把标签数据转为ID（以数量为上限）
label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
# label2idx = dict(zip(uniqueLabel, uniqueLabel))
print('label2idx:{}'.format(label2idx))
labelList = list(range(len(uniqueLabel)))
print('labelList:{}'.format(labelList))

labelIds = [label2idx[label] for label in lableIDs]
print('labelIds:{}'.format(labelIds))

trainLabels = np.array(labelIds[:2], dtype="float32")
print('trainLabels:{}'.format(trainLabels))

