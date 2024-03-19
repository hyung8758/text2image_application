"""
LogUtil: manage logging methods.

Hyungwon Yang
24.03.06
MediaZen
"""

import os
import logging
from datetime import datetime

class LogUtils:

    # get log config
    @classmethod
    def init_log(cls, log_path: str,
                    log_file_name: str = "main.log",
                    log_in_date: bool = False,
                    log_format: str = "%(asctime)s(%(module)s:%(lineno)d)%(levelname)s:%(message)s") -> str:

        # make log dir.
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        # make date dir if necessary.
        if log_in_date:
            # get current date
            date_info = cls.get_date()
            cur_date_dir = os.path.join(log_path, date_info)
            if not os.path.exists(cur_date_dir):
                os.mkdir(cur_date_dir)
            log_path = cur_date_dir
        
        # set config info.
        log_file_path = os.path.join(log_path, log_file_name)
        logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file_path)
        logging.info("log path: {}".format(log_file_path))
        logging.info("START logging...")
        return log_path
            
    @classmethod
    def get_date(cls, date_format:str ="%y%m%d"):
        return datetime.now().strftime(date_format)
        
    @staticmethod
    def info(input_str: str):
        now = datetime.now()
        date_format = now.strftime("%Y-%m-%d %H:%M:%S")
        print("{} INFO: {}".format(date_format, input_str))
    
    @staticmethod
    def warning(input_str: str):
        now = datetime.now()
        date_format = now.strftime("%Y-%m-%d %H:%M:%S")
        print("{} WARNING: {}".format(date_format, input_str))

    @staticmethod
    def error(input_str: str):
        now = datetime.now()
        date_format = now.strftime("%Y-%m-%d %H:%M:%S")
        print("{} ERROR: {}".format(date_format, input_str))

