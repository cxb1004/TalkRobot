import configparser
import os


class Config:
    """
    静态代码，类初始化的时候运行，对所有实例来说只运行一次
    """
    try:
        # 拼接获得config.ini路径
        __CONFIG_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        __CONFIG_FILE_NAME = 'config.ini'
        # 读入配置文件
        __cf = configparser.RawConfigParser()
        __cf.read(os.path.join(__CONFIG_FILE_PATH, __CONFIG_FILE_NAME))
        print('读入config.ini配置：' + __cf.get('version', 'name'))
    except Exception as e:
        print("载入配置文件失败: " + os.path.join(__CONFIG_FILE_PATH, __CONFIG_FILE_NAME))
        print(e)

    def get_value(self, section, option):
        try:
            value = self.__cf.get(section, option)
            return value
        except Exception as e:
            print("配置文件中没有该配置内容: section[" + section + "] option: " + option)
            raise e

    def get_project_root_dir(self):
        return str(self.get_value('project', 'root_folder'))
