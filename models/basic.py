'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: basic.py
@time: 1/1/19 9:58 PM
@desc:
'''

from .cpm_vgg16 import cpm_vgg16
def obtain_model(configure, points):
  if configure.arch == 'cpm_vgg16':
    net = cpm_vgg16(configure, points)
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net