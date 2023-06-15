
import torch
import torch.nn as nn
import numpy as np
import os
import models.cifar as models
import resnet


def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)

class EAR(nn.Module):

    def __init__(self, locc):
        super(EAR, self).__init__()
        self.num_experts = len(locc)
        self.experts = nn.ModuleList([models.__dict__['resnet'](num_classes=100, depth=110, block_name='BasicBlock') for i in range(self.num_experts)])
        # load the weights for each experts
        basename_ = 'F:/Research/PHD_AIZU/tiny_ai/ear/checkpoint_experts/ear_wts/r-110-subset/'
        
        for i, wt in enumerate(locc):
          list_of_classes = wt
          wt_name = basename_
          for i, loc in enumerate(list_of_classes):
              wt_name += str(loc)
              if (i < len(list_of_classes)-1):
                  wt_name += '_'
          
          #wt_name = str(a) + '_' + str(b) + '_' + str(c) + '.pth.tar'
          wt_name += '.pth.tar' 
          wt_ = torch.load(wt_name)
          self.experts[i].load_state_dict(wt_)
        
        self.prouter = models.__dict__['resnet'](num_classes=100, depth=20, block_name='BasicBlock') 
        chk = torch.load('F:/Research/PHD_AIZU/tiny_ai/ear/ck_backup/cifar100/resnet-depth-20/checkpoint/model_best.pth.tar')
        self.prouter.load_state_dict(chk['state_dict'])

       
        self.router = models.__dict__['resnet'](num_classes=self.num_experts, depth=110, block_name='BasicBlock')
        self.router = nn.Sequential(
            self.router,
            nn.Sigmoid()
        )
  
        chk = torch.load('F:/Research/PHD_AIZU/tiny_ai/ear/router_wts/r110.pth.tar')
        self.router.load_state_dict(chk)
       
        #assert(self.k <= self.num_experts)

    def forward(self, x, thres=0.5):

        '''
        we do not need to predict all the relevant experts correctly.
        We just need a single expert to be prediction correctly for inference.
        '''
        rout = self.router(x)
        rout_temp = rout
        rout_temp = rout_temp.detach()
        
        rout = rout.detach().cpu().numpy()
        preds = np.array(rout > thres, dtype=float)
        
        #exp_output = [self.experts[i](x) for i, j in enumerate(preds[0]) if (j)]
        exp_output = []
        rout_sorted = torch.argsort(rout_temp, dim=1, descending=True)
        if (preds[0][rout_sorted[0][0]]):
           exp_output.append(self.experts[rout_sorted[0][0]](x))
        exp_output.append(self.prouter(x))
        avg_output = average(exp_output)
        return avg_output

# confusing_classes = [[35, 98], [55, 72], [47, 52], [11, 35], [11, 46], [70, 92], [13, 81], [47, 96], [2, 35], [81, 90], [52, 59], [62, 92], [78, 99], [5, 25], [30, 95], [50, 74], [30, 73], [10, 61], [33, 96], [44, 78], [67, 73], [23, 71], [46, 98], [52, 96], [2, 11], [35, 46], [13, 58], [18, 44], [26, 45], [4, 55]]
# net = EAR(confusing_classes)
# in_ = torch.rand((1, 3, 32, 32))
# input_ = torch.rand(1,3, 32, 32)
# out_ex = net(input_)
# print (out_ex.shape)
# print (out_ex[0].shape)
# # print (out_r)
# # print (out_r[:, 0])
# # val = out_r[:, 0]
# # val = torch.unsqueeze(val, 1)
# # print (val)
# #print (out_ex[0] * val)
# #out_ex[0] *= 0
# # lst = torch.cat(out_ex, 0)
# # print ("expert out shape", lst.shape)
# # rt = torch.transpose(out_r, 0, 1)
# # rt = out_r
# # print ("router shape", rt.shape)
# # print (torch.mul(rt, lst))
      
