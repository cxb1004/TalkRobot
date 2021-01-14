import pymysql
import warnings
from common.config import Config
from common.log import Log

warnings.filterwarnings("ignore")

config = Config()
log = Log()


class Database:
    __connect = None
    __cursor = None

    def __init__(self):
        if self.__connect is None or self.__cursor is None:
            self.connect()

    def isConnected(self):
        if self.__connect is not None:
            return True
        return False

    def connect(self):
        # 连接数据库
        host = config.get_value('db', 'host')
        port = config.get_value('db', 'port')
        database = config.get_value('db', 'database')
        user = config.get_value('db', 'user')
        password = config.get_value('db', 'password')
        charset = config.get_value('db', 'charset')

        if self.__connect is None:
            try:
                self.__connect = pymysql.connect(host=host, port=int(port), database=database, user=user,
                                                 password=password, charset=charset)
                log.debug('连接成功！')
            except:
                log.error(
                    '连接失败, 请查看连接参数：{} | {} | {} | {} | {} | {}'.format(host, port, database, user, password, charset))
                exit(999)

        if self.__cursor is None:
            self.__cursor = self.__connect.cursor()

    def close(self):
        # 手动释放数据库资源
        if self.__cursor is not None:
            self.__cursor.close()
        if self.__connect is not None:
            self.__connect.close()
        log.debug('连接关闭！')

    def execute(self, sql, params):
        try:
            self.__cursor.execute(sql, params)
            self.__connect.commit()
        except Exception as e:
            self.__connect.rollback()
            raise e

    def query(self, sql, params):
        try:
            self.__cursor.execute(sql, params)
            result = self.__cursor.fetchall()
            return result
        except Exception as e:
            raise e


# 测试代码
# db = Database()
# try:
#     sql = 'SELECT * FROM bigdata.robot_industry where id=%s;'
#     params = [1]
#     result = db.query(sql, params)
#     print(result)
#
#     sql = 'SELECT * FROM bigdata.robot_industry where id=%s;'
#     params = [2]
#     result = db.query(sql, params)
# finally:
#     db.close()
#
# print(result)