# coding=utf-8
# aurthor: luruiyaun
# date: 2017-9-27
# file: logger.py

__all__ = ['get_logger', 'set_logger', 'debug', 'info', 'warning', 'error', 'critical']

import logging
from logging import Formatter
import threading


DEFAULT_FMT      = '[%(levelname)s] [%(asctime)s] %(filename)s [line:%(lineno)d] %(message)s'
DEFAULT_DATE_FMT = '%Y-%m-%d %a, %p %H:%M:%S'
DEFAULT_LEVEL    = 'DEBUG'

def get_logger(loggername='', cmdlog=True, filelog=True, filename='myApp.log', filemode='a', colorful=True,
                cmd_color_dict=None, cmdlevel='DEBUG', cmdfmt=DEFAULT_FMT, cmddatefmt=DEFAULT_DATE_FMT,
                filelevel='INFO', filefmt=DEFAULT_FMT, filedatefmt=DEFAULT_DATE_FMT,
                backup_count=0, limit=10240, when=None):

    return Logger.get_logger(**locals())


class CmdColor():
    ''' Cmd color escape strings '''
    # color escape strings
    __COLOR_RED    = '\033[1;31m'
    __COLOR_GREEN  = '\033[1;32m'
    __COLOR_YELLOW = '\033[1;33m'
    __COLOR_BLUE   = '\033[1;34m'
    __COLOR_PURPLE = '\033[1;35m'
    __COLOR_CYAN   = '\033[1;36m'
    __COLOR_GRAY   = '\033[1;37m'
    __COLOR_WHITE  = '\033[1;38m'
    __COLOR_RESET  = '\033[1;0m'

    # color names to escape strings
    __COLOR_2_STR = {
        'red'   : __COLOR_RED,
        'green' : __COLOR_GREEN,
        'yellow': __COLOR_YELLOW,
        'blue'  : __COLOR_BLUE,
        'purple': __COLOR_PURPLE,
        'cyan'  : __COLOR_CYAN,
        'gray'  : __COLOR_GRAY,
        'white' : __COLOR_WHITE,
        'reset' : __COLOR_RESET,
    }

    __COLORS = __COLOR_2_STR.keys()
    __COLOR_SET = set(__COLORS)

    @classmethod
    def get_color_by_str(cls, color_str):
        if not isinstance(color_str, str):
            raise TypeError("color string must str, but type: '%s' passed in." % type(color_str))
        color = color_str.lower()
        if color not in cls.__COLOR_SET:
            raise ValueError("no such color: '%s'" % color)
        return cls.__COLOR_2_STR[color]

    @classmethod
    def get_all_colors(cls):
        ''' return a list that contains all the color names '''
        return cls.__COLORS

    @classmethod
    def get_color_set(cls):
        ''' return a set contains the name of all the colors'''
        return cls.__COLOR_SET


class BasicFormatter(Formatter):

    def __init__(self, fmt=None, datefmt=None):
        super(BasicFormatter, self).__init__(fmt, datefmt)
        self.default_level_fmt = '[%(levelname)s]'

    def formatTime(self, record, datefmt=None):
        ''' @override logging.Formatter.formatTime
            default case: microseconds is added
            otherwise: add microseconds mannually'''
        asctime = Formatter.formatTime(self, record, datefmt=datefmt)
        return asctime if datefmt is None or datefmt == '' else self.default_msec_format % (asctime, record.msecs)

    def format(self, record):
        ''' @override logging.Formatter.format
            generate a consistent format'''
        msg = Formatter.format(self, record)
        pos1 = self._fmt.find(self.default_level_fmt) # return -1 if not find
        pos2 = pos1 + len(self.default_level_fmt)
        if pos1 > -1:
            last_ch = self.default_level_fmt[-1]
            repeat = self._get_repeat_times(msg, last_ch, 0, pos2)
            pos1 = self._get_index(msg, last_ch, repeat)
            return '%-10s%s' % (msg[:pos1], msg[pos1+1:])
        else:
            return msg

    def _get_repeat_times(self, string, sub, start, end):
        cnt, pos = 0, start
        while 1:
            pos = string.find(sub, pos)
            if pos >= end or pos == -1:
                break
            cnt += 1
            pos += 1
        return cnt

    def _get_index(self, string, substr, times):
        pos = 0
        while times > 0:
            pos = string.find(substr, pos) + 1
            times -= 1
        return pos


class CmdColoredFormatter(BasicFormatter):
    '''Cmd Colored Formatter Class'''

    # levels list and set
    __LEVELS = ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    __LEVEL_SET = set(__LEVELS)

    def __init__(self, fmt=None, datefmt=None, **level_colors):
        super(CmdColoredFormatter, self).__init__(fmt, datefmt)
        self.LOG_COLORS = {}     # a dict, used to convert log level to color
        self.init_log_colors()
        self.set_level_colors(**level_colors)

    def init_log_colors(self):
        ''' initialize log config '''
        for lev in CmdColoredFormatter.__LEVELS:
            self.LOG_COLORS[lev] = '%s'

    def set_level_colors(self, **kwargs):
        ''' set each level different colors '''
        lev_set = CmdColoredFormatter.__LEVEL_SET
        color_set = CmdColor.get_color_set()

        # check log level and set colors
        for lev, color in kwargs.items():
            lev, color = lev.upper(), color.lower()
            if lev not in lev_set:
                raise KeyError("log level '%s' does not exist" % lev)
            if color not in color_set:
                raise ValueError("log color '%s' does not exist" % color)
            self.LOG_COLORS[lev] = ''.join([CmdColor.get_color_by_str(color), '%s', CmdColor.get_color_by_str('reset')])

    def format(self, record):
        ''' @override BasicFormatter.format'''
        msg = super(CmdColoredFormatter, self).format(record)
        # msg = BasicFormatter.format(self, record)     # 本行和上一行等价
        return self.LOG_COLORS.get(record.levelname, '%s') % msg


class Logger():
    ''' My logger '''
    # log related arguments
    __LOG_ARGS = ['cmdlog', 'cmd_color_dict', 'filelog', 'filename', 'filemode', 'colorful', 'cmdlevel','loggername',
                'cmdfmt', 'cmddatefmt', 'filelevel', 'filefmt', 'filedatefmt', 'backup_count', 'limit', 'when']
    __log_arg_set = set(__LOG_ARGS)
    __lock = threading.Lock()
    __name2logger = {}

    @classmethod
    def _acquire_lock(cls):
        cls.__lock.acquire()
    
    @classmethod
    def _release_lock(cls):
        cls.__lock.release()

    @classmethod
    def get_logger(cls, **kwargs):
        loggername = kwargs['loggername']
        if loggername not in cls.__name2logger:
            cls._acquire_lock()    # lock current thread
            if loggername not in cls.__name2logger:
                log_obj = object.__new__(cls)
                cls.__init__(log_obj, **kwargs)
                cls.__name2logger[loggername] = log_obj
            cls._release_lock()    # release lock
        return cls.__name2logger[loggername]

    def set_logger(self, **kwargs):
        ''' Configure logger with dict settings '''
        for k, v in kwargs.items():
            if k not in Logger.__log_arg_set:
                raise KeyError("config argument '%s' does not exist" % k)
            setattr(self, k, v) # add instance attributes

        if self.cmd_color_dict is None:
            self.cmd_color_dict = {'debug': 'green', 'warning':'yellow', 'error':'red', 'critical':'purple'}
        if isinstance(self.cmdlevel, str):
            self.cmdlevel = getattr(logging, self.cmdlevel.upper(), logging.DEBUG)
        if isinstance(self.filelevel, str):
            self.filelevel = getattr(logging, self.filelevel.upper(), logging.INFO)

        self.__init_logger()
        self.__import_log_func()
        if self.cmdlog:
            self.__add_streamhandler()
        if self.filelog:
            self.__add_filehandler()

    def __init__(self, **kwargs):
        self.logger = None
        self.streamhandler = None
        self.filehandler = None
        self.set_logger(**kwargs)

    def __init_logger(self):
        ''' Init logger or reload logger '''
        if self.logger is None:
            self.logger = logging.getLogger(self.loggername)
        else:
            logging.shutdown()
            self.logger.handlers.clear()

        self.streamhandler = None
        self.filehandler = None
        self.logger.setLevel(DEFAULT_LEVEL)

    def __import_log_func(self):
        ''' Add common functions into current class'''
        func_names = ['debug', 'info', 'warning', 'error', 'critical', 'exception']
        for fn in func_names:
            f = getattr(self.logger, fn)
            setattr(self, fn, f)

    def __add_filehandler(self):
        ''' Add a file handler to logger '''
        # Filehandler
        if self.backup_count == 0:
            self.filehandler = logging.FileHandler(self.filename, self.filemode)
        # RotatingFileHandler
        elif self.when is None:
            self.filehandler = logging.handlers.RotatingFileHandler(self.filename,
                                    self.filemode, self.limit, self.backup_count)
        # TimedRotatingFileHandler
        else:
            self.filehandler = logging.handlers.TimedRotatingFileHandler(self.filename,
                                        self.when, 1, self.backup_count)

        formatter = BasicFormatter(self.filefmt, self.filedatefmt)
        self.filehandler.setFormatter(formatter)
        self.logger.addHandler(self.filehandler)

    def __add_streamhandler(self):
        ''' Add a stream handler to logger '''
        self.streamhandler = logging.StreamHandler()
        self.streamhandler.setLevel(self.cmdlevel)
        formatter = CmdColoredFormatter(self.cmdfmt, self.cmddatefmt,
                    **self.cmd_color_dict) if self.colorful else BasicFormatter(self.cmdfmt, self.cmddatefmt)
        self.streamhandler.setFormatter(formatter)
        self.logger.addHandler(self.streamhandler)


if __name__ == '__main__':
    print("logger测试")
    log = get_logger(colorful=True)
    # log.set_logger(colorful=False)
    log.debug('原谅绿')
    log.info('info白')
    log.warning("提高log等级到warning, loggername为'log'")
    log.set_logger(cmdlevel='warning', loggername='log')
    log.info('不存在的一句话')
    log.warning("我怎么黄了!!!")
    log.error("我的天哪")
    log.critical("红得发紫!!!!!!!")
    log.warning("修改log等级为debug")
    log.set_logger(cmdlevel='debug')
    log.debug("修改debug颜色配置为灰色")
    log.set_logger(cmd_color_dict={'debug':'gray'})
    log.debug('修改完成')

    print("同名时单例模式测试")
    log.set_logger(cmdlevel='error')

    # log = get_logger()
    log.debug("呵呵大1")
    log.debug("调整cmd输出format为: %s" % '%a, %H:%M:%S')
    log2 = get_logger(cmddatefmt='%a, %H:%M:%S')
    log2.debug("呵呵大2")
    log.debug("id 检测: log:%s log2:%s" % (id(log), id(log2)))
    log.debug("相等性检测: log is log2 %s" % (log is log2))
    print("不同名时非单例测试")
    log3 = get_logger(loggername='test_logger3')
    log3.debug("呵呵大3")
    log3.warning("相等性检测, log 应出现两条，因为Root和log3都会打印一次: log is log3: %s | log2 is log3 %s" % (log is log3, log2 is log3))