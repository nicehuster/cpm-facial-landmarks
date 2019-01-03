'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: saver.py
@time: 1/1/19 10:35 PM
@desc:
'''

import torch

def save_checkpoint(state, filename, logger):
  torch.save(state, filename)
  logger.log('save checkpoint into {}'.format(filename))
  return filename