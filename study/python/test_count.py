# from collections import Counter
#
# #统计词频
# colors = ['red', 'blue', 'red', 'green', 'blue', 'blue']
#
# result = Counter(colors)
# print(result)
# print(list(result.elements()))
#
# sortWordCount = sorted(result.items(), key=lambda item: item[0], reverse=True)
# print(sortWordCount)
import numpy as np
vocab = []
# wordEmbedding是词向量，即用200维的数字数组来表示一个词
wordEmbedding = []

# 分词后不在词典内的词经常被标为<UNK>，处理为相同长度通常会在前或后补<PAD>
# PAD：使用无损方法采用0向量   UNK(unknown):一般采用随机向量
vocab.append("PAD")
vocab.append("UNK")
wordEmbedding.append(np.zeros(200))
wordEmbedding.append(np.random.randn(200))
print(vocab)
print(wordEmbedding)
