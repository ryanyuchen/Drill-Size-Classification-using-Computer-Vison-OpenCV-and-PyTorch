import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
Reference:
    https://arxiv.org/pdf/1901.05555.pdf
    https://arxiv.org/abs/1708.02002
    https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
    https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
    
    https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
    https://github.com/kaidic/LDAM-DRW/blob/3193f05c1e6e8c4798c5419e97c5a479d991e3e9/losses.py#L7
    
    https://github.com/mbsariyildiz/focal-loss.pytorch/blob/master/focalloss.py
    https://github.com/laughtervv/Deeplab-Pytorch/blob/master/models/losses.py
'''

def reweight(cls_num_list, beta=0.9999):
    '''
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    '''
    per_cls_weights = None
    # Reference: https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights)

    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        '''
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        '''
        loss = None
        # Reference: https://github.com/kaidic/LDAM-DRW/blob/3193f05c1e6e8c4798c5419e97c5a479d991e3e9/losses.py#L13
        ce_inputs = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p = torch.exp(-ce_inputs)
        loss_array = (1 - p) ** self.gamma * ce_inputs
        loss = loss_array.mean()

        return loss