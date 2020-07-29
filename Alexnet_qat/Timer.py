import time
import logging
from config import cfg
class Timer_logger:
    def __init__(self):
        self.start_time = 0
        logging.basicConfig(filename=cfg.logger_path+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.log',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            level=logging.INFO
                            )

    def start(self):
        self.start_time = time.time()

    def log(self):
        time_info = time.time() - self.start_time
        # logging.basicConfig(filename='logger.log', level=logging.INFO)
        logging.info("Timer log: ")
        logging.info(time_info)

    def log_info(self,string):
        logging.info(string)


