import math
import random

import numpy as np
import torch
from torch.autograd import Variable


class Helper(object):
    """
        工具类： 用来提供各种工具函数
    """
    def __init__(self):
        self.timber = True
        self._preDict = {}


    def np2var(self, x, type=torch.FloatTensor):
        """
        :param x(numpy data), type (if we use the embedding layer, type should be torch.LongTensor)
        :return: y(Variable)
        """
        return Variable(torch.Tensor(x, type=type))

    def init_hidden(self, batch, hidden_size, num_layers=1):
        h_0 = Variable(torch.zeros(batch, num_layers, hidden_size))
        c_0 = Variable(torch.zeros(batch, num_layers, hidden_size))
        return h_0, c_0

    def to_var(self, x, use_gpu):
        x = Variable(x)
        if use_gpu:
            x = x.cuda()
        return x

    def expend(self, data, hashtag_num):
        return data.view(data.data.size()[0], 1, data.data.size()[1]).expand(data.data.size()[0], hashtag_num, data.data.size()[1])

    def expend_u(selfself, data, hashtag_num):
        return data.view(data.data.size()[0],1).expand(data.data.size()[0],hashtag_num)
    # Protocol: leave-1-out evaluation
    # Measures: Recall and NDCG

    def count_Recall(self, pred_y, tags, true_list, k=1):
        """
        :param pred_y: FloatTensor size: [101]
        :param tags:  LongTensor size: [101]
        :param true_label: int size: 1
        :param k: int
        """
        value, ranklist = torch.topk(pred_y, k)
        ranklist2 = torch.index_select(tags, 0, torch.LongTensor(ranklist.cpu()))
        ranklist2 = ranklist2.cpu().numpy()

        pre_str=''
        for j in ranklist2:
            pre_str+=j+', '
        pre_str=pre_str[:-2]+'\n'
        with open('case-pred.txt', 'a', encoding='utf-8') as f:
            f.write(pre_str)

        for j in ranklist2:
            for true_l in true_list:
                if j == true_l:
                    return 1
        return 0
    def count_ndcg(self, pred_y, tags, true_list, k=1):
        """
        :param pred_y: FloatTensor size: [101]
        :param tags:  LongTensor size: [101]
        :param true_label: int size: 1
        :param k: int
        """
        value, ranklist = torch.topk(pred_y, k)
        count = 0
        ranklist2 = torch.index_select(tags, 0, torch.LongTensor(ranklist.cpu()))
        ranklist2 = ranklist2.cpu().numpy()

        for j in ranklist2:
            for true_l in true_list:
                if j == true_l:
                    return(math.log(2) / math.log(count+2))
            count += 1
        return 0

def gen_A(num_classes, adj_matrix):

    _adj=adj_matrix * 0.25 /(adj_matrix.sum(0)+ 1e-6)
    _adj=_adj+np.identity(num_classes,np.int)
    return _adj

def gen_adj(A):

    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)


    adj = torch.mm(torch.mm(A, D).t(), D)
    return adj


