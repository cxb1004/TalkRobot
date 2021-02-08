import os
import re

from common.log import Log

log = Log()


def removeFileIfExists(filePath):
    """
    如果文件已经存在，就删除文件
    :param filePath:
    :return:
    """
    if os.path.isfile(filePath):
        os.remove(filePath)
        log.debug('已经删除存在的文件{}'.format(filePath))


def replaceMutiSpace(str):
    """
    多个空格替换成单个空格
    :param str:
    :return:
    """
    str = re.sub(' +', ' ', str)
    return str

# str = '123   456'
# print(replaceMutiSpace(str))