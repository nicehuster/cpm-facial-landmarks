'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: prepare_seed.py
@time: 1/1/19 7:17 PM
@desc:
'''
import numpy as np
import torch,random

def prepare_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)#为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(rand_seed)#为当前GPU设置随机种子；
    torch.cuda.manual_seed_all(rand_seed)#为所有的GPU设置种子