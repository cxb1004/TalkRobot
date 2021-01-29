import logging.handlers
from common.config import Config


class Log:
    config = Config()

    __DEFAULT_LOGGER_NAME = config.get_value('log', 'default_logger_name')
    __DEFAULT_LOGGER_FORMAT = config.get_value('log', 'default_logger_format')
    __DEFAULT_LOGGER_LEVEL = logging.getLevelName(config.get_value('log', 'default_logger_level'))
    __LOGGER_LOCATION = config.get_value('log', 'default_logger_logfile')
    __LOGGER_WHEN = config.get_value('log', 'default_logger_when')
    __LOGGER_INTERVAL = int(config.get_value('log', 'default_logger_interval'))
    __LOGGER_BACKUP_COUNT = int(config.get_value('log', 'default_logger_backup_count'))
    __LOGGER_ENCODING = config.get_value('log', 'default_logger_encoding')
    """
    静态初始化
    """
    # 内置日志
    __default_log = logging.getLogger(__DEFAULT_LOGGER_NAME)
    # 默认日志配置（日志格式、日志等级）
    __default_formatter = logging.Formatter(__DEFAULT_LOGGER_FORMAT)
    __default_log.setLevel(__DEFAULT_LOGGER_LEVEL)
    # 默认往控制台输出日志
    __console = logging.StreamHandler()
    __console.setLevel(__DEFAULT_LOGGER_LEVEL)
    __console.setFormatter(__default_formatter)
    __default_log.addHandler(__console)
    #
    __fileByDateHandle = logging.handlers.TimedRotatingFileHandler(filename=__LOGGER_LOCATION,
                                                                   when=__LOGGER_WHEN,
                                                                   interval=__LOGGER_INTERVAL,
                                                                   backupCount=__LOGGER_BACKUP_COUNT,
                                                                   encoding=__LOGGER_ENCODING)
    __fileByDateHandle.setLevel(__DEFAULT_LOGGER_LEVEL)
    __fileByDateHandle.setFormatter(__default_formatter)
    __default_log.addHandler(__fileByDateHandle)

    def info(self, msg):
        self.__default_log.info(msg)

    def debug(self, msg):
        self.__default_log.debug(msg)

    def warn(self, msg):
        self.__default_log.warning(msg)

    def error(self, msg, exception):
        self.__default_log.error(msg)
        self.__default_log.error(exception)

    def error(self, msg):
        self.__default_log.error(msg)
