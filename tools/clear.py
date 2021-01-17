"""
1、获得当前目录
2、获得当前目录下所有的文件
3、循环读取文件
4、按行读取每个文件的内容
5、读取第一个非空字符
5.1 以#开头的行：删除整行
5.2 以''' 或 \""" 开头的行: 删除当前行以及之后的行，直到再出现一次''' 或 \"""
"""

import sys
import warnings
import os

sys.path.append("..")
# from common1.log import ProjectLog
# from common1.config import Config as ProjConfig

warnings.filterwarnings("ignore")
# log = ProjectLog()
# projConfig = ProjConfig()
# ROOT_PATH = projConfig.get_project_root_dir()
CUR_PATH = os.path.abspath('.')
CUR_FILE = 'clear.py'

CHAR1 = '#'
CHAR2 = "'''"
CHAR3 = '"""'

print('程序开始运行...')


def startWith(s1, s2):
    return s1.strip().startswith(s2)


def transform(newFile, oldFile):
    with open(oldFile, 'r', encoding='utf-8', newline='') as rFile:
        with open(newFile, 'w', encoding='utf-8', newline='') as wFile:

            clear_line_flag = False
            lines = rFile.readlines()
            for line in lines:
                if clear_line_flag == True:
                    wLine = None
                    if startWith(line, CHAR2) or startWith(line, CHAR3):
                        clear_line_flag = False
                else:
                    if startWith(line, CHAR2) or startWith(line, CHAR3):
                        wLine = None
                        clear_line_flag = True
                    elif startWith(line, CHAR1):
                        wLine = None
                    else:
                        wLine = line

                if not wLine is None:
                    wFile.write(wLine)


for root, dirs, files in os.walk(CUR_PATH):
    path = root
    for file in files:
        if not (file == CUR_FILE or file == 'cmd'):
            oldFile = path + os.sep + file
            newFile = path + os.sep + 'backup_' + file
            os.rename(oldFile, newFile)
            transform(oldFile, newFile)
            print('文件转化完毕：{}'.format(oldFile))

print('程序结束运行')
