"""

"""
import os
import sys

# 当前目录
basePath = os.path.abspath(os.path.dirname(__file__))
# 设置当前目录为执行运行目录
sys.path.append(basePath)

from common.log import Log

log = Log()


"""
主程序

文件说明
【data/labeledTrainData.csv】：读入文件，是打标签的数据，包括content/label_id/label
【data/corpus.txt】：输出文件，用于生成词向量文件
"""
# 读入数据：打过标签的数据文件 content / label_id / label
csv_labeled_data = os.path.join(basePath, 'data/labeledTrainData.csv')
# 如果文件不存在就返回错误
if not os.path.isfile(csv_labeled_data):
    log.error('labeledTrainData.csv文件不存在，程序无法执行，请检查！')
    exit(999)

txt_corpus = os.path.join(basePath, 'data/corpus.txt')

vec_size = 100
vec_window = 6
vec_min_count = 2
vec_sg = 1
vec_iter = 5
model = gensim.models.Word2Vec(sentences,
                               size=vec_size,
                               window=vec_window,
                               min_count=vec_min_count,
                               sg=vec_sg,
                               iter=vec_iter)
log.info('生成中...')
model.wv.save_word2vec_format(word2vec_file, binary=True)
log.info('生成完毕')









