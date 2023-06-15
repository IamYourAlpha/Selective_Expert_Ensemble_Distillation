from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

class Logits(nn.Module):
    ''' Make logit distillation '''
    def __init__(self):
        super(Logits, self).__init__()
    
    def forward(self, out_student, out_teacher):
        loss = F.mse_loss(out_student, out_teacher)
        return loss


class SoftTarget(nn.Module):
    def __init__(self, teacher_temp=5, student_temp=3):
        super(SoftTarget, self).__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def forward(self, out_student, out_teacher):
        loss = F.kl_div(F.log_softmax(out_student/self.student_temp, dim=1),
                        F.softmax(out_teacher/self.teacher_temp, dim=1), reduction='batchmean') * self.teacher_temp * self.student_temp
        return loss
#                         reduction='batchmean'