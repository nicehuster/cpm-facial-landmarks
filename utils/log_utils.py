'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: log_utils.py
@time: 1/1/19 7:18 PM
@desc:
'''
import time
from pathlib import Path
def time_for_file():
  ISOTIMEFORMAT='%d-%h-at-%H-%M-%S'
  return '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))

def time_string():
  ISOTIMEFORMAT='%Y-%m-%d %X'
  string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

def print_log(print_string, log):
  if isinstance(log, Logger): log.log('{:}'.format(print_string))
  else:
    print("{:}".format(print_string))
    if log is not None:
      log.write('{:}\n'.format(print_string))
      log.flush()
def convert_secs2time(epoch_time, return_str=False):
  need_hour = int(epoch_time / 3600)
  need_mins = int((epoch_time - 3600*need_hour) / 60)
  need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
  if return_str:
    str = '[Time Left: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
    return str
  else:
    return need_hour, need_mins, need_secs


class Logger(object):

  def __init__(self, log_dir, logstr):
    """Create a summary writer logging to log_dir."""
    self.log_dir = Path(log_dir)
    self.model_dir = Path(log_dir) / 'checkpoint'
    self.meta_dir = Path(log_dir) / 'metas'
    self.log_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
    self.model_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
    self.meta_dir.mkdir(mode=0o775, parents=True, exist_ok=True)

    self.logger_path = self.log_dir / '{:}.log'.format(logstr)
    self.logger_file = open(str(self.logger_path), 'w')

  def __repr__(self):
    return ('{name}(dir={log_dir})'.format(name=self.__class__.__name__, **self.__dict__))

  def path(self, mode):
    if mode == 'meta':
      return self.meta_dir
    elif mode == 'model':
      return self.model_dir
    elif mode == 'log':
      return self.log_dir
    else:
      raise TypeError('Unknow mode = {:}'.format(mode))

  def last_info(self):
    return self.log_dir / 'last-info.pth'

  def extract_log(self):
    return self.logger_file

  def close(self):
    self.logger_file.close()

  def log(self, string, save=True):
    print(string)
    if save:
      self.logger_file.write('{:}\n'.format(string))
      self.logger_file.flush()