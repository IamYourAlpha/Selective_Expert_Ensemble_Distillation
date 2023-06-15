# -*- coding: utf-8 -*-
"""
@author: intisar chowdhury
inst: The University of Aizu.
"""

# In[]:
from __future__ import print_function
import argparse

# torch
import csv
from genericpath import exists
from os import path
from os.path import basename
from typing_extensions import final
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
#from torch.optim import optimizer
from torchvision import datasets
from losses.kd import Logits, SoftTarget#, transforms
import transforms
from torch.autograd import Variable
from torch.utils.data import  SubsetRandomSampler, WeightedRandomSampler
import os
import random
import time
import json
import copy
import numpy as np
import pandas as pd
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import StepLR
# models
import models.cifar as models
# import multi-branch/coupled MS-NET
from utils.ms_net_utils import *
from utils.data_utils import *
from utils.model_utils import *
from losses import *
from tqdm import tqdm

########### GPU statistics ###########################
use_cuda = torch.cuda.is_available()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print ("[INFO] List of GPUS are: {}".format(available_gpus))
print ("[INFO] Total number of GPUS avaiable for training: {}".format(torch.cuda.device_count()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Selective Ensemble of Expert Distillation')

# Hyper-parameters
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=128, type=int, metavar='N',
                    help='test batchsize')


parser.add_argument('--schedule', type=int, nargs='+', default=[10],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--train_end_to_end', action='store_true',
                    help='train from router to experts')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--router_epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--corrected_images', type=str, default='./corrected_images/')
###############################################################################
parser.add_argument('--expert_epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train experts')
##########################################################################
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--initialize_with_router', action='store_true', default=True)

parser.add_argument('--cuda', action='store_true', default=True,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=10, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--evaluate_only_router', action='store_true',
                    help='evaluate router on testing set')

parser.add_argument('--weighted_sampler', action='store_true',
                    help='what sampler you want?, subsetsampler or weighted')

parser.add_argument('--finetune_experts', action='store_true', default=True,
                    help='perform fine-tuning of layer few layers of experts')
parser.add_argument('--save_images', action='store_true', default=True)
###########################################################################
parser.add_argument('--train_mode', action='store_true', default=True, help='Do you want to train or test?')

parser.add_argument('--alpha_prob', type=int, default=50, help='alpha probability')
 


############################################################################

parser.add_argument('--topk', type=int, default=3, metavar='N',
                    help='topn?')
parser.add_argument('--experts', type=int, default=100, metavar='N',
                    help='how many experts you want?')
parser.add_argument('--id', type=str)

###########################################################################

parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

parser.add_argument('-d', '--dataset', default='cifar100', type=str)


# Architecture details
parser.add_argument('--arch', '-a', metavar='ARCH', default='preresnet',
                    help='backbone architecture')
parser.add_argument('--depth', type=int, default=8, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
                    help='initial learning rate to train')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-gpu', '--gpu_id', default=0, type=str, help='set gpu number')


#########################
    # Random Erasing
    # Turns it on if you wish to boost performance a bit like 1%
parser.add_argument('--p', default=0.3, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.3, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
state = str(state)
with open('state.txt', 'w') as f:
    f.write(state)
f.close()
model_weights = {}

classes = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
           'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

cifar10_LABELS_LIST = ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# class_rev = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
#            4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

svhn_LABELS_LIST = ['zero', 'one', 'two', 'three',
           'four', 'five', 'six', 'seven', 'eight', 'nine']
# class_rev = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
#            4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


class_rev = {0: 'zero', 1: 'one', 2: 'two', 3: 'three',
           4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}


cifar100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

label_list = {'cifar10' : ['airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], 'cifar100': [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'],
    'svhn': ['zero', 'one', 'two', 'three',
           'four', 'five', 'six', 'seven', 'eight', 'nine'],
    'fmnist': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
           'Coat', 'Sandal', 'Shirt', 'Sneaker', 'bag', 'Ankle-boot']
}

if (use_cuda):
    args.cuda = True
# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y/T, dim=1), 
                        F.softmax(teacher_scores/T, dim=1)) \
                        * (T*T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

def make_npy(vectors, name):
    basefolder = './vectors'
    if (not os.path.exists(basefolder)):
        os.mkdir(basefolder)
    filename = os.path.join(basefolder, name)
    np.save(filename, vectors)

def get_bernouli_list(alpha_prob):
    bernouli = []
    for i in range(alpha_prob):
        bernouli.append(1) # Pr(beta) == X.X
    for i in range(alpha_prob, 100):
        bernouli.append(0) 
    return bernouli

def train(epoch,
          model,
          experts,
          teacher,
          Q_loi, 
          train_loader, 
          train_loader_all_data, 
          optimizer, 
          teacher_temp=3, 
          student_temp=5, 
          stocastic_loss=False
          ): 
    
    model.train()
    
    for k, v in experts.items():
        if (k in Q_loi):
            print ("Changing mode of experts:", k)
            experts[k].eval()
    teacher.eval()
    loss_fn = SoftTarget(teacher_temp=teacher_temp, student_temp=student_temp)
    #loss_fn = Logits()
    alpha = 1.0 # 0.9999 # 0.999 # 0.99 # 0.9 
    output_teacher = []
    for batch_idx, (dta, target) in enumerate(train_loader):
        output_teacher = []
        if args.cuda:
            dta, target = dta.to(device), target.to(device)
        dta, target = Variable(dta), Variable(target)
        
        optimizer.zero_grad()
        output, _ = model(dta)

        for exp_ in Q_loi:
            temp_out, _ = experts[exp_](dta)
            output_teacher.append(temp_out)
        #output_teacher = [(experts[Q](dta)) for Q in Q_loi]
        
        teacher_output, _ = teacher(dta)
        output_teacher.append(teacher_output)
        output_teacher = average(output_teacher)
        output_teacher = output_teacher.detach()
        loss_kd = loss_fn(output, output_teacher) * alpha
        loss_ce = F.cross_entropy(output, target) * (1. - alpha)
        loss = loss_kd + loss_ce
        loss.backward()
        optimizer.step()

        if (batch_idx % 100 == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(dta), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.item()))
       
    return model, optimizer

def test(model, 
         test_loader, 
         best_so_far, 
         name, 
         print_acc=False, 
         save_wts=False, 
         save_vectors=False,
         test_expert=True
         ):
    
    model.eval()
    test_loss = 0
    correct = 0
    found_best = False
    vectors = []
    vectors_teacher_signal = []
    vectors_predicted_signal = []
    for dta, target in test_loader:
        if args.cuda:
            dta, target = dta.to(device), target.to(device)
        dta, target = Variable(dta, volatile=True), Variable(target)
        output, feature_vector = model(dta)
        output = F.softmax(output, dim=1)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = torch.argsort(output, dim=1, descending=True)[0:, 0]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # save feature vectors etc for vis.
        vectors.extend(feature_vector.detach().cpu().numpy())
        vectors_teacher_signal.extend(target.detach().cpu().numpy())
        vectors_predicted_signal.extend(pred.detach().cpu().numpy())
        
    test_loss /= len(test_loader.dataset)
    
    if (save_vectors):
        vectors = np.array(vectors)
        vectors_teacher_signal = np.array(vectors_teacher_signal)
        vectors_predicted_signal = np.array(vectors_predicted_signal)
        print ("The vectors shape of features: {}".format(vectors.shape))
        print ("The vectors shape of teacher signal: {}".format(vectors_teacher_signal.shape))
        print ("The vectors shape of predicted signal: {}".format(vectors_predicted_signal.shape))
        
        name_npy = name + '.npy'
        name_npy_teacher = name + '_teacher' + '.npy'
        name_npy_predicted = name + '_predicted' + '.npy'
        make_npy(vectors, name_npy)
        make_npy(vectors_teacher_signal, name_npy_teacher)
        make_npy(vectors_predicted_signal, name_npy_predicted)

    if (print_acc and (not test_expert)):
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct.double() / len(test_loader.dataset) ))
    
    elif (print_acc and  test_expert):
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct.double() / 100.00 ))
        
        
    if (not save_wts):
        correct = correct.double()
        return None, correct
    
    now_correct = correct.double()
    
    if best_so_far < now_correct:
        print ("best correct: ", best_so_far)
        print ("now correct: ", now_correct)
        found_best = True
        wts = copy.deepcopy(model.state_dict()) # deep copy        
        name += ".pth.tar"
        save_checkpoint(wts, found_best, name)
        best_so_far = now_correct
        
    return best_so_far, now_correct 

def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)
    
def inference_with_experts_and_routers(test_loader, experts, router, loe_with_wts, original_=True, topk=2):

    """ function to perform evaluation with experts
    params:
    ------
    test_loader: data loader for testing dataset
    experts: dictionary of expert Neural Networks
    router: router network
    topK: upto how many top-K you want to re-check?
    returns:
    -------
    whole system accuracy --> router + experts
    """
    freqMat = np.zeros((100, 100)) # -- debug
    router.eval()
    experts_on_stack = []
    expert_count = {} 
    for k, v in experts.items():
        experts[k].eval()
        experts_on_stack.append(k)
        expert_count[k] = 0
    
    count = 0
    counter_to_save_conf = 0
    ext_ = '.png'
    correct = 0
    by_experts, by_router = 0, 0
    mistake_by_experts, mistake_by_router = 0, 0
    agree, disagree = 0, 0

    for dta, target in tqdm(test_loader):
        count += 1
        # if (count == 500):
        #     break
        if args.cuda:
            dta, target = dta.to(device), target.to(device)
        dta, target = Variable(dta, volatile=True), Variable(target)
        output_raw, _ = router(dta)
        output = F.softmax(output_raw, dim=1)
        router_confs, router_preds = torch.sort(output, dim=1, descending=True)
        preds = []
        confs = []
        for k in range(0, topk):
            #ref = torch.argsort(output, dim=1, descending=True)[0:, k]
            ref = router_preds[0:, k]
            conf = router_confs[0:, k]
            preds.append(ref.detach().cpu().numpy()[0]) # simply put the number. not the graph
            confs.append(conf.detach().cpu().numpy()[0])
    
        #cuda0 = torch.device('cuda:0')
        experts_output = []
        router_confident = False
        # for exp_ in experts_on_stack:
        #     if (topk == 3):
        #         if (str(preds[0]) in exp_ and str(preds[1]) in exp_ and str(preds[2]) in exp_): # TOP-3
        #             router_confident = False
        #             break
        #     else:
        #         if (str(preds[0]) in exp_ and str(preds[1]) in exp_): # TOP-2
        #             router_confident = False
        #             break
        
        list_of_experts = []
        target_string = str(target.cpu().numpy()[0])

        # Check for the type of inference you want

        # for pred__ in preds:
        #     in_ = target_string in str(pred__)
        #     if (in_):
        #         print (in_)
        #         list_of_experts.append(str(pred__))
        #         print ("Lenghth of list : {}".format(len(list_of_experts)))
        
        # activates only for the single class MS-NET!
        # exists_ = (target_string in str(preds[0])) or \
        #               (target_string in str(preds[1])) or \
        #               (target_string in str(preds[2]))    
        
        for exp in loe_with_wts:#experts_on_stack: 
            #exists_ = target_string in exp # )):# original  
            if (topk == 3):
                #(not router_confident) and exists_ and
                if ((str(preds[0]) in exp) or (str(preds[1]) in exp) or (str(preds[2]) in exp)): # top-3
                    router_confident = False
                    list_of_experts.append(exp)
                    expert_count[exp] += 1
                    #break
            if (topk == 2): #exists_ and 
                router_confident = False
                if ( (str(preds[0]) in exp) or (str(preds[1]) in exp)):
                    router_confident = False
                    list_of_experts.append(exp)
                    expert_count[exp] += 1
                    #break      

        if (router_confident):
            if (preds[0] == target.cpu().numpy()[0]):
                correct += 1
                by_router += 1
                continue
            else:
                mistake_by_router += 1            
        # else:
        #     for exp in experts_on_stack: #and
        #         if(topk == 3):
        #             if ( (str(preds[0]) in exp and str(preds[1]) in exp and str(preds[2]) in exp)):
        #                 #list_of_experts.append(exp)
        #                 #expert_count[exp] += 1
        #                 break
        #         if (topk == 2):
        #             if ( (str(preds[0]) in exp and str(preds[1]) in exp)):                            
        #                 #list_of_experts.append(exp)
        #                 #expert_count[exp] += 1
        #                 break
           
        #experts_output = [experts[exp_](dta) for exp_ in list_of_experts] # prediction from all experts
        
        for exp_ in list_of_experts:
            temp_out, _ = experts[exp_](dta)
            experts_output.append(temp_out)

        experts_output.append(output_raw) # append the router raw logits
        if (len(experts_output) > 1):
            experts_output_avg = average(experts_output) # normalize by the router confidence
        else:
            experts_output_avg = output_raw
        experts_output_prob = F.softmax(experts_output_avg, dim=1) # F.softmax(output_raw, dim=1)
        router_output_prob = F.softmax(output_raw, dim=1)
        #pred = torch.argsort(experts_output_prob, dim=1, descending=True)[0:, 0]
        exp_conf, exp_pred = torch.sort(experts_output_prob, dim=1, descending=True)
        pred, conf_ = exp_pred[0:, 0], exp_conf[0:, 0]
        # Check if experts prediction correct or not
        if (pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
            correct += 1
            by_experts += 1
        else:
            freqMat[pred.cpu().numpy()[0]][target.cpu().numpy()[0]]  += 1
            freqMat[target.cpu().numpy()[0]][pred.cpu().numpy()[0]]  += 1
            mistake_by_experts += 1
        # count if experts and router agrees
        if (pred.cpu().numpy()[0]  == preds[0] \
            and pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
            agree += 1
        # count how many times they disagree.
        elif (pred.cpu().numpy()[0]  != preds[0]\
                and pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
            disagree += 1
            final_pred, final_conf =  pred.detach().cpu().numpy()[0], conf_.detach().cpu().numpy()[0]
            
            if (counter_to_save_conf < 0):
                name_of_dataset = '%s'%args.dataset
                print ("*" * 50)
                print ("Experts confidence")
                experts_conf = []
                for i in range(len(label_list[name_of_dataset])):
                    experts_conf.append(experts_output_prob.detach().cpu().numpy()[0][i])
                    #print (experts_output_prob.detach().cpu().numpy()[0][i], end=" ")
                print ("")
                print ("Router confidence")
                router_conf = []
                for i in range(len(label_list[name_of_dataset])):
                    router_conf.append(router_output_prob.detach().cpu().numpy()[0][i])
                    #print (router_output_prob.detach().cpu().numpy()[0][i], end=" ")
                # print (final_pred, final_conf)
                # print ("*" * 50)

                barchart(label_list['%s'%args.dataset],
                        experts_conf,
                        router_conf,
                        rects1_label='Experts confidence',
                        rects2_label='Router confidence',
                        y_labels='softmax')
                counter_to_save_conf += 1
    
            # Save misclassified samples
            args.save_images = False
            if (args.save_images):
                data_numpy = dta[0].cpu() # transfer to the CPU.
                f_name = '%d'%count + '%s'%ext_ # set file name with ext
                f_name_no_text = '%d'%count + 'no_text' + '%s'%ext_
                if (not os.path.exists(args.corrected_images)):
                    os.makedirs(args.corrected_images)
                imshow(data_numpy, os.path.join(args.corrected_images, f_name), \
                    os.path.join(args.corrected_images, f_name_no_text), \
                    fexpertpred=class_rev[final_pred], fexpertconf=final_conf, \
                        frouterpred=class_rev[preds[0]], frouterconf=confs[0])
                    
            
    print ("Router and experts agrees with {} samplers \n\n and router and experts disagres for {}".format(agree, disagree))
    print ("Total Samples corrected by the Experts: {}".format(disagree))        
    #print ("Routers: {} \n Experts: {}".format(by_router, by_experts))
    print ("Mistakes by Routers: {} \n Mistakes by Experts: {}".format(mistake_by_router, mistake_by_experts))
    #print (expert_count)
    #print (correct)
    return correct, freqMat, disagree, expert_count


def ensemble_inference(test_loader, experts, Q_loi, router):
    '''
    Perform inference using all the available experts.
    '''
    router.eval()
    for k, v in experts.items():
        if (k in Q_loi):
            experts[k].eval()
    correct = 0
    test_loss = 0
    for dta, target in test_loader:
        all_outputs=  []
        if args.cuda:
            dta, target = dta.to(device), target.to(device)
        dta, target = Variable(dta, volatile=True), Variable(target)
        output, _ = router(dta)
        #all_outputs = [experts[Q](dta) for Q in Q_loi]

        for exp_ in Q_loi:
            temp_out, _ = experts[exp_](dta)
            all_outputs.append(temp_out)


        all_outputs.append(output)
        all_outputs_avg = average(all_outputs)
        all_output_prob = F.softmax(all_outputs_avg)
        output_final = F.softmax(all_output_prob)
        test_loss += F.cross_entropy(output_final, target).item() # sum up batch loss
        pred = torch.argsort(output_final, dim=1, descending=True)[0:, 0]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    
    print('\nEnsemble Performance: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
         100. * correct.double() / len(test_loader.dataset) ))

def adjust_learning_rate(epoch, optimizer):
    if epoch in args.schedule:
        print ("\n\n***************CHANGED LEARNING RATE TO\n*********************\n")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        
        for param in optimizer.param_groups:
            print ("Lr {}".format(param['lr']))

def main():

    global best_acc
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    # same router as the ms-net, o-ms-net.
    train_loader_router, test_loader_router, num_classes, test_loader_single = prepare_dataset_for_router(args.dataset,
                                                                                                          args.train_batch,
                                                                                                          args.test_batch)
    print("==================> creating model...")
    
    router, roptimizer = make_router_and_optimizer(num_classes,
                                                   args.dataset,
                                                   args.arch,
                                                   args.depth,
                                                   args.block_name,
                                                   args.learning_rate,
                                                   load_weights=True)

    size_of_router = sum(p.numel() for p in router.parameters() if p.requires_grad == True)
    print ("Network size {:.2f}M".format(size_of_router/1000000))
    #########################################################################
    # Heatmap for the confusing classes
    #print ("Calculating the heatmap for confusing class .....\n")
    print ("[INFO:] Calculating an rough assumption on difficult pairs of classes, please wait ..")
    matrix = np.array(calculate_matrix(router, test_loader_single, num_classes, args.cuda, only_top2=False, topk=args.topk), dtype=int)
    #####################################################################
    
    ls = np.arange(num_classes)
    #heatmap(matrix, ls, ls) # show heatmap
    
    ####################################################################################################
    #matrix_args, values = return_topk_args_from_heatmap(matrix, num_classes, args.experts, binary_=True, topk=args.topk)
    confusing_classes, values = return_topk_args_from_heatmap(matrix, num_classes, args.experts, binary_=True, topk=args.topk)
    

    
    values = None
    #confusing_classes_str = get_confusing_classes_str(confusing_classes)
    confusing_classes_str = []
    for sub in confusing_classes:
        index = ""
        for i, sb in enumerate(sub):
            index += str(sb)
            if (i < len(sub)-1):
                index += "_"
        confusing_classes_str.append(index)
    print (confusing_classes_str)   
  
    print ("*"*50)
    if (args.dataset =='cifar10'):
        matrix_args = [[num_] for num_ in range(0, 10)] #num_classes
    if (args.dataset =='cifar100'):
        matrix_args = [[num_] for num_ in range(0, 100)] #num_classes
    print (matrix_args)
    print ("*"*50)

    # unique_classes = set()
    # labels = []
    # for mat, val  in zip(matrix_args, values):
    #     temp_list = []
        
    #     if (args.topk == 2):
    #         labels.append(str(mat[0]) + "," + str(mat[1]))
    #         unique_classes.add(mat[0])
    #         unique_classes.add(mat[1])
    #         print ("TOP-1: {}, TOP-2: {}, # of occurance: {}".format(label_list['%s'%args.dataset][mat[0]], label_list['%s'%args.dataset][mat[1]], val))
        
    #     if (args.topk == 3):
    #         print ("TOP-1: {}, TOP-2: {}, # of occurance: {}".format(label_list['%s'%args.dataset][mat[0]], label_list['%s'%args.dataset][mat[1]], val))
    #         labels.append(str(mat[0]) + "," + str(mat[1]) + "," + str(mat[2]))
    #         unique_classes.add(mat[0])
    #         unique_classes.add(mat[1])
    #         unique_classes.add(mat[2])
    # print ("*"*50)
    # print ("Total unique classes with the {} pairs: {}".format(args.experts, len(unique_classes)))

    expert_train_dataloaders,  expert_test_dataloaders, lois = prepeare_dataset_for_experts(args.dataset,
                                                                                            matrix_args,
                                                                                            values,
                                                                                            args.train_batch,
                                                                                            args.test_batch,
                                                                                            weighted_sampler=True)
    experts, eoptmizers = load_expert_networks_and_optimizers(lois,
                                                              num_classes, 
                                                              args.dataset,
                                                              args.arch,
                                                              args.depth,
                                                              args.block_name,
                                                              initialize_with_router=True,
                                                              finetune_experts=True)
    
    teacher = load_teacher_network() # just load it incase you need it for better performnce.
    # you can load any teacher network you want to boost performance of experts.
    
    print ("External teacher loaded")
    
    args.evaluate_only_router = False
    
    if (args.evaluate_only_router):
        experts = load_state_dict_for_experts(lois,
                                experts,
                                args.dataset, 
                                args.arch, 
                                args.depth,
                                first_init=True,
                                path='./checkpoint_experts/pre-trained/'
                                #path='./checkpoint_experts/wts_100/exp_15/top-2/'
                                )
        test(router, test_loader_router, best_so_far=None, name='_', print_acc=True, save_wts=False, save_vectors=False)
        ensemble_inference(test_loader_router, experts, lois, router)
        return
    
    router, _ =         make_router_and_optimizer(num_classes, # tot. number of classes
                                                   args.dataset, # the dataset name -> cifar10, cifar100, svhn..
                                                   args.arch, # networks, supports resnet series only
                                                   args.depth, # depth of resnet, e.g. 8, 20, 32, 56, 110.
                                                   args.block_name,
                                                   args.learning_rate,
                                                   load_weights=True)
    
    indexes=['_test_experts', '_test_all']
    plot = {}
    plots, lst = make_list_for_plots(lois, plot, indexes)       
    co_ = 0
    Q_loi = [elem for elem in lois]
    
    args.train_mode = False
    
    if (args.train_mode):
        best_so_far = {}
        for loi in lois:
            best_so_far[loi] = 0.0
        garbage = 99999999
        experts = load_state_dict_for_experts(lois,
                                experts,
                                args.dataset, 
                                args.arch, 
                                args.depth,
                                first_init=True,
                                path='./checkpoint_experts/pre-trained/'
                                #path='./checkpoint_experts/wts/exp_23/top-2/'
                                )
        start = time.time()
        
        while(True):
            if (co_ == 100):
                break
            co_ = co_ + 1
            current_expert = Q_loi.pop(0)
            Q_common = set()
            #put common element in Q_common/overlapping with current_experts.
            # for elem in lois:
            #     if (elem == current_expert):
            #         continue
            #     else:
            #         if (current_expert[0] in elem or current_expert[2] in elem or current_expert[4] in elem):
            #             Q_common.append(elem)
            Q_temp = confusing_classes_str
      
            if (args.topk == 2):
                 for elem in Q_temp:
                     p1, p2 = elem.split('_')
                     if (current_expert == p1):
                         Q_common.add(p2)
                     elif (current_expert == p2):
                        Q_common.add(p1)
            
            if (args.topk == 3):
                for elem in Q_temp:
                    p1, p2, p3 = elem.split('_')
                    if (current_expert == p1):
                        Q_common.add(p2)
                        Q_common.add(p3)
                    elif (current_expert == p2):
                        Q_common.add(p1)
                        Q_common.add(p3)
                    elif (current_expert == p3):
                        Q_common.add(p1)
                        Q_common.add(p2)
      
            Q_common = list(Q_common)
            print ("\n Q_common ID: {}".format(Q_common)) 
            expert_names = ""
            for qc in Q_common:
                expert_names += label_list[args.dataset][int(qc)]
                expert_names += ", "
            print ("Q_common names: {}".format(expert_names))
            #print ("Q_LOI: {}".format(Q_loi))
            
            print ("\n Current expert ID: {}".format(current_expert))
            print ("Current expert name: {}".format(label_list[args.dataset][int(current_expert)]))
            
            experts = load_state_dict_for_experts(Q_common,
                                        experts,
                                        args.dataset, 
                                        args.arch, 
                                        args.depth,
                                        first_init=False,
                                        path='./checkpoint_experts/wts_100/exp_21/top-2/'
                                        )
       
            # Reset the optimizer 
            eoptmizers = reset_optimizer(lois, experts)
            print ("Now iter: {}".format(co_))
            args.expert_epochs = 15
            for epoch in range(0, args.expert_epochs):
                adjust_learning_rate(epoch, eoptmizers[current_expert])
                experts[current_expert], eoptmizers[current_expert] = train(epoch,
                                                                            experts[current_expert], # the current expert from QUEUE
                                                                            experts, # all the experts
                                                                            router, # teacher
                                                                            Q_common, # the QUEUE of list of rest of experts/name of experts
                                                                            expert_train_dataloaders[current_expert], # train loader for corresponding expert
                                                                            train_loader_router, # Common train loader for the router network
                                                                            eoptmizers[current_expert], # optimizer for corresponding expert
                                                                            teacher_temp=1, # temp for experts
                                                                            student_temp=1,
                                                                            stocastic_loss=True # Do you want MS-NET LOSS
                                                                            )
                
                best_so_far[current_expert], test_acc_on_expert_data = test(experts[current_expert],
                                                                            expert_test_dataloaders[current_expert],
                                                                            best_so_far[current_expert], 
                                                                            current_expert,
                                                                            print_acc=True, 
                                                                            save_wts=True,
                                                                            save_vectors=False,
                                                                            test_expert=True
                                                                            ) # save to queuer
                _, c = test(experts[current_expert],
                            test_loader_router,
                            garbage,
                            current_expert,
                            print_acc=True, 
                            save_wts=False,
                            save_vectors=False
                            )
                
                ensemble_inference(expert_test_dataloaders[current_expert], experts, Q_common, router)
              
            
            Q_loi.append(current_expert)
            for k, v in experts.items():
                experts[k].train()
        ''' naming convention:
        numberOfexperts_typeofexperts_w/woKD
        '''
        #filename = 'oracle_resnet110_stocasticloss_fine_tuning_weightedsampler_cifar_100.csv'
    #filename = 'r110_svhn_random_init_subset.csv'
    #to_csv(plots, filename)
    router, roptimizer = make_router_and_optimizer(num_classes,
                                                   args.dataset,
                                                   args.arch,
                                                   args.depth,
                                                   args.block_name,
                                                   args.learning_rate,
                                                   load_weights=True)
    #end_ = time.time()
    #print(f"Runtime of the program is {end_ - start}")
    
    print ("*" * 50)
    best_so_far = 0
    base_location = 'checkpoint_experts'#temp'#/T5A8W3
    pth_folder = 'wts_100/exp_21/top-2/'
    
    # This is for the diss report
    #pth_folder = 'diss/teacher-init/%s/top-%s/resnet-%s'%(args.dataset, args.topk, args.depth)
    
    pth_exists = False
    tot_load = 0
    loe_with_wts = []
    print (os.path.join(base_location, pth_folder))

    for i in tqdm(range(0, len(lois))):
        loi = lois[i]
        # _, temp_r_sub = test(router,
        #             expert_test_dataloaders[loi],
        #             best_so_far,
        #             "router" + loi,
        #             print_acc=False,
        #             save_wts=False,
        #             save_vectors=False
        # )
        
        # _, temp_r_all = test(router,
        #             test_loader_router,
        #             best_so_far,
        #             "router" + loi,
        #             print_acc=False,
        #             save_wts=False,
        #             save_vectors=False
        #             )

        # print ("\n \n Performance of ROUTER in SUB classes [{}] : {}".format(loi, temp_r_sub))
        # print ("Performance of ROUTER in ALL classes [{}] : {}".format(loi, temp_r_all))

        wts_loc = os.path.join(base_location, pth_folder, '%s'%loi + '.pth.tar')
        pth_exists = os.path.exists(wts_loc)
        if (pth_exists):
            wts = torch.load(wts_loc)
            tot_load += 1
            pth_exists = False
            loe_with_wts.append(loi)
        else:
            continue
        #wts = torch.load(os.path.join(base_location, pth_folder, '%s'%loi + '.pth.tar'))
        #wts = torch.load('./checkpoint_experts/infer/5.pth.tar')
        #print ("{} \n \n ".format("*" * 50))   
        
        experts[loi].load_state_dict(wts)
        # _, temp_exp_sub = test(experts[loi],
        #                 expert_test_dataloaders[loi],
        #                 best_so_far,
        #                 loi,
        #                 print_acc=False,
        #                 save_wts=False,
        #                 save_vectors=False
        #                 )
        # _, temp_exp_all = test(experts[loi],
        #                 test_loader_router,
        #                 best_so_far,
        #                 loi,
        #                 print_acc=False,
        #                 save_wts=False,
        #                 save_vectors=False
        #                 )
        
        # print ("\n \n Performance of EXPERT in SUB classes [{}] : {}".format(loi, temp_exp_sub))
        # print ("Performance of EXPERT in ALL classes [{}] : {}".format(loi, temp_exp_all))
        # print ("{} \n \n ".format("*" * 50))
    
    print ("Total loaded experts: {}".format(tot_load))
    #ensemble_inference(test_loader_router, experts, router)
    print ("Setting up to perform inference with experts and routers .... \n")
    topk = args.topk
    original_ = False
    #ensemble_inference(test_loader_router, experts, lois, router)
    print ("List of experts with wts: {}".format(loe_with_wts))
    accuracy_exp, m, corrected_samples, expert_count = inference_with_experts_and_routers(test_loader_single,
                                                                                          experts,
                                                                                          router,
                                                                                          loe_with_wts,
                                                                                          original_,
                                                                                          topk)
    #heatmap(m, ls, ls)
    expert_count_values_list = []
    for k, v in expert_count.items():
        expert_count_values_list.append(expert_count[k])
    
    # barchart(labels,
    #         values,
    #         expert_count_values_list,
    #         rects1_label='ICC Count',
    #         rects2_label='Frequency of experts invoked',
    #         y_labels='counts'
    #         )

    _, accuracy_router = test(router, test_loader_router, best_so_far, "router", save_wts=False)
    print ("Router ACC: {} \n Experts: {}\n".format(accuracy_router, accuracy_exp))
    print ("## Actual performance of experts with router: {}".format(accuracy_router + corrected_samples))


if __name__ == '__main__':
    main()
# %%