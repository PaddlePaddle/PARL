#-*- coding: utf-8 -*-
#File: test_logging.py

from parl.utils import logger
import threading as th

logger.set_level('info')
logger.set_dir('./test_dir')

logger.debug('debug')
logger.info('info')
logger.warn('warn')
logger.error('error')


def thread_func():
    logger.info('test thread')


th_list = []
for i in range(10):
    t = th.Thread(target=thread_func)
    t.start()
    th_list.append(t)

for t in th_list:
    t.join()
