# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:39:14 2021

@author: intisar
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import cv2, torch
from torchvision.transforms.transforms import Resize
from itertools import permutations

__all__ = ['return_topk_args_from_heatmap', 'heatmap', 'save_checkpoint', 'calculate_matrix',\
           'make_list_for_plots', 'to_csv', 'imshow', 'barchart']

def sort_by_value(dict_, reverse=False):
    
    dict_tuple_sorted = {k: v for k, v in \
                         sorted(dict_.items(),\
                                reverse=True, key=lambda item: item[1])}
    return dict_tuple_sorted

def get_comb(x1, x2, x3, x4, x5):
    '''
    get combinations of n-dim. list

    params:
    -------
    x1, x2, x3,... --> list of numbers

    returns:
    --------
    all possible compbination of x1, x2, .... 
    
    '''
    perm = permutations([x1, x2, x3, x4, x5], 5)
    return perm


def return_topk_args_from_heatmap(matrix, n, no_of_experts, binary_=True, topk=2):
    ''' 
    params:
    ------ 
    matrix(int, int): interclass correlation matrix
    n (int): No of classes in dataset
    no_of_experts (int): How many experts do you want, or how many subsets?
    binary_  (bool) =   
    topk: upto how many prediction you want  to evaluate?


    return:
    ------
    new_tuple: tuples of classes that are confusing to the router network
    value_of_tuple: Value corresponding to each tuples.
    '''


    if (topk == 2):
        visited = np.zeros((n,n))
        tuple_list = []
        value_of_tuple = []
        dict_tuple = {}
        for i in range(0, n):
            for j in range(0, n):
                if ( not visited[i][j] and not visited[j][i] and matrix[i][j]>0):
                    dict_tuple[str(i) + "_" + str(j)] =  matrix[i][j]
                    visited[i][j] = 1
                    visited[j][i] = 1
                        
        dict_sorted = sort_by_value(dict_tuple, reverse=True)    
        for k, v in dict_sorted.items():
            sub_1, sub_2 = k.split("_")
            sub_1, sub_2 = int(sub_1), int(sub_2)
            tuple_list.append([sub_1, sub_2])
            value_of_tuple.append(v)
            if (len(tuple_list) == no_of_experts):
                break
        return tuple_list, value_of_tuple

    
    elif (topk == 3):
        visited = np.zeros((n,n, n))
        tuple_list = []
        value_of_tuple = []
        dict_tuple = {}
        for i in range(0, n):
            for j in range(0, n):
                for k in range(0, n):
                    if ((not visited[i][j][k] and not visited[i][k][j] and not visited[j][i][k] and not visited[j][k][i] and \
                        not visited[k][i][j] and not visited[k][j][i] )\
                            and matrix[i][j][k]>0 and (i!=j and j!=k and i!=k)):
                        dict_tuple[str(i) + "_" + str(j) + "_" + str(k)] =  matrix[i][j][k]
                       
                        visited[i][j][k] = 1
                        visited[i][k][j] = 1

                        visited[j][i][k] = 1
                        visited[j][k][i] = 1

                        visited[k][i][j] = 1
                        visited[k][j][i] = 1

       
        dict_sorted = sort_by_value(dict_tuple, reverse=True)   
        for k, v in dict_sorted.items():
            sub_1, sub_2, sub_3 = k.split("_")
            sub_1, sub_2, sub_3 = int(sub_1), int(sub_2), int(sub_3)
            tuple_list.append([sub_1, sub_2, sub_3])
            value_of_tuple.append(v)
            if (len(tuple_list) == no_of_experts):
                break
        return tuple_list, value_of_tuple
    
    elif (topk == 5):
        print ("top-5")
        visited = np.zeros((n, n, n, n, n), dtype='uint8')
        tuple_list = []
        value_of_tuple = []
        dict_tuple = {}

        for i in range(0, n):
            for j in range(0, n):
                if (i != j):
                    for k in range(0, n):
                        if (k != j and k != i):
                            for l in range(0, n):
                                if (l != i and l != j and l != k):
                                    for m in range(0, n):
                                        if (m != i and m != j and m != k and m != l):
                                            if (not visited[i][j][k][l][m] and matrix[i][j][k][l][m] > 0):
                                                dict_tuple[str(i) + "_" + str(j) + "_" + str(k) + "_" + str(l) + "_" + str(m)] =  matrix[i][j][k][l][m]
                                                print (i, j, k, l , m)
                                                list_of_comb = get_comb(i, j , k, l, m)
                                                for comb in list_of_comb:
                                                    x1, x2, x3, x4, x5 = comb
                                                    if (not visited[x1][x2][x3][x4][x5]):
                                                        visited[x1][x2][x3][x4][x5] = 1
        
        dict_sorted = sort_by_value(dict_tuple, reverse=True)   
        for k, v in dict_sorted.items():
            sub_1, sub_2, sub_3, sub_4, sub_5 = k.split("_")
            sub_1, sub_2, sub_3, sub_4, sub_5 = int(sub_1), int(sub_2), int(sub_3), int(sub_4), int(sub_5)
            tuple_list.append([sub_1, sub_2, sub_3, sub_4, sub_5])
            value_of_tuple.append(v)
            if (len(tuple_list) == no_of_experts):
                break
        return tuple_list, value_of_tuple



    #     binary_ = True
    #     if (binary_):
    #         return tuple_list, value_of_tuple
    #     ''' if not binary we go further and 
    #     return more expert classes
    #     '''
    #     new_tuple = tuple_list.copy()
    #     for keys in tuple_list:
    #         key1, key2 =  keys
    #         temp1 = set()
    #         temp2 = set()
    #         temp1.add(key1)
    #         temp1.add(key2)
    #         temp2.add(key2)
    #         temp2.add(key1)
    #         for keys2 in tuple_list:
    #             if (key1 in keys2):
    #                 first_elem, second_elem = keys2
    #                 temp1.add(first_elem)
    #                 temp1.add(second_elem)

    #         for keys2 in tuple_list:
    #             if (key2 in keys2):
    #                 first_elem, second_elem = keys2
    #                 temp1.add(first_elem)
    #                 temp1.add(second_elem)
    #         if (len(temp1)>2 and (list(temp1) not in tuple_list and list(temp1) not in new_tuple)):
    #             new_tuple.append(list(temp1))
    #         if (len(temp2)>2 and (list(temp2) not in tuple_list and list(temp2) not in new_tuple)):
    #             new_tuple.append(list(temp2))

    # return new_tuple, value_of_tuple

    
    #return tuple_list, value_of_tuple


def barchart(
    labels, inter_class_correlation,
    frequent_experts,
    rects1_label='ICC Count',
    rects2_label='Frequency of experts invoked',
    y_labels='counts'
    ):
    """ Calculates the barcharts for ICC and most frequent count of experts on test set
    """
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, inter_class_correlation, width, label=rects1_label)
    rects2 = ax.bar(x + width/2, frequent_experts, width, label=rects2_label)
    ax.set_ylabel(y_labels)
    ax.set_title('')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.xticks(rotation=45)
    #if ( ('confidence' in rects1_label) or ('confidence' in rects2_label)):
    #    plt.xticks(fontsize=5)
    plt.show()
    

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
   

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False) 
    plt.show()
    if not os.path.exists('checkpoint/figures/'):
        os.makedirs('checkpoint/figures/')
    #figure_name = 'checkpoint/figures/heatmap_%s.png'%str(depth)
    #plt.savefig(figure_name)
    return im, cbar

def make_list_for_plots(lois, plot, indexes):
    lst = []
    for loi in lois:
        for index in indexes:
            ind = loi + index
            lst.append(ind)
            plot[ind] = []
    
    return plot, lst

def calculate_matrix(model, test_loader_single, num_classes, cuda, only_top2=True, topk=2):
    model.eval()
    stop_at = 500
    tot = 0
    
    if (topk == 2):
        freqMat = np.zeros((num_classes, num_classes))
        for dta, target in test_loader_single:
            if cuda:
                dta, target = dta.cuda(), target.cuda()
            # tot = tot + 1    
            # if (tot == stop_at):
            #     break
            with torch.no_grad():
                output, _ = model(dta)
                output = F.softmax(output, dim=1)
                pred1 = torch.argsort(output, dim=1, descending=True)[0:, 0]
                pred2 = torch.argsort(output, dim=1, descending=True)[0:, 1]
                if (only_top2):
                    if (pred2.cpu().numpy()[0] == target.cpu().numpy()[0]):
                        s = pred2.cpu().numpy()[0]
                        d = pred1.cpu().numpy()[0]
                        freqMat[s][d] += 1
                        freqMat[d][s] += 1
                else:
                    if (pred1.cpu().numpy()[0] != target.cpu().numpy()[0]):
                        s = pred1.cpu().numpy()[0]
                        d = target.cpu().numpy()[0]
                        freqMat[s][d] += 1
                        freqMat[d][s] += 1

    elif (topk == 3):
        freqMat = np.zeros((num_classes, num_classes, num_classes))
        for dta, target in test_loader_single:
            # tot = tot + 1    
            # if (tot == stop_at):
            #     break
            if cuda:
                dta, target = dta.cuda(), target.cuda()
            with torch.no_grad():
                output, _ = model(dta)
                output = F.softmax(output, dim=1)
                pred1 = torch.argsort(output, dim=1, descending=True)[0:, 0]
                pred2 = torch.argsort(output, dim=1, descending=True)[0:, 1]
                pred3 = torch.argsort(output, dim=1, descending=True)[0:, 2]
            
                if (pred2.cpu().numpy()[0] == target.cpu().numpy()[0] or pred3.cpu().numpy()[0] == target.cpu().numpy()[0]):
                    s1 = pred1.cpu().numpy()[0]
                    s2 = pred2.cpu().numpy()[0]
                    s3 = pred3.cpu().numpy()[0]

                    temp_set = set()
                    for pd1 in [s1, s2, s3]:
                        temp_set.add(pd1)
                        for pd2 in [s1, s2, s3]:
                            temp_set.add(pd2)
                            for pd3 in [s1, s2, s3]:
                                temp_set.add(pd3)
                                if (len(temp_set) == 3):
                                    freqMat[pd1][pd2][pd3] += 1
                                    temp_set = set()
                                    
    elif (topk == 5):
        freqMat = np.zeros((num_classes, num_classes, num_classes, num_classes, num_classes), dtype='uint8')
        for dta, target in test_loader_single:
            # tot = tot + 1    
            # if (tot == stop_at):
            #     break
            if cuda:
                dta, target = dta.cuda(), target.cuda()

            # tot = tot + 1 
            # if (tot == stop_at):
            #     break
            with torch.no_grad():
                output, _ = model(dta)
                output = F.softmax(output, dim=1)
                pred1 = torch.argsort(output, dim=1, descending=True)[0:, 0]
                pred2 = torch.argsort(output, dim=1, descending=True)[0:, 1]
                pred3 = torch.argsort(output, dim=1, descending=True)[0:, 2]
                pred4 = torch.argsort(output, dim=1, descending=True)[0:, 3]
                pred5 = torch.argsort(output, dim=1, descending=True)[0:, 4]
                
                if (pred2.cpu().numpy()[0] == target.cpu().numpy()[0] or pred3.cpu().numpy()[0] == target.cpu().numpy()[0] or \
                    pred4.cpu().numpy()[0] == target.cpu().numpy()[0] or pred5.cpu().numpy()[0] == target.cpu().numpy()[0]):
                    s1 = pred1.cpu().numpy()[0]
                    s2 = pred2.cpu().numpy()[0]
                    s3 = pred3.cpu().numpy()[0]
                    s4 = pred4.cpu().numpy()[0]
                    s5 = pred5.cpu().numpy()[0]
                    temp_set = set()
                    demo = [s1, s2, s3, s4, s5]
                    for pd1 in [s1, s2, s3, s4, s5]:
                        for pd2 in [s1, s2, s3, s4, s5]:
                            if (pd2 != pd1):
                                for pd3 in [s1, s2, s3, s4, s5]:
                                    if (pd3 != pd2 and pd3 != pd1):
                                        for pd4 in [s1, s2, s3, s4, s5]:
                                            if (pd4 != pd3 and pd4 != pd2 and pd4!=pd1):
                                                for pd5 in [s1, s2, s3, s4, s5]:
                                                    if (pd5 != pd4 and pd5!=pd3 and pd5!=pd2 and pd5!=pd1):
                                                        freqMat[pd1][pd2][pd3][pd4][pd5] += 1
                                                       
                                                        
    print ("done")    
    return freqMat


def imshow(img, f_name, f_name_no_text, fexpertpred=None, fexpertconf=None, frouterpred=None, frouterconf=None):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy() # pytorch tensor is usually (n, C(0), H(1), W(2))
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # numpy needs (H(1), W(2), C(0))
    plt.savefig(f_name_no_text)
    pred = str(fexpertpred)
    conf = str(fexpertconf)
    pred_r = str(frouterpred)
    pred_c = str(frouterconf)
    res = "Experts prediction: " + pred + " : " + conf + "\n" + "Routers prediction: " + pred_r + ":" + pred_c
    #plt.text(0,0, res, color='r')
    plt.text(0, 0, res, style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.savefig(f_name)
    plt.close()
    
def save_checkpoint(model_weights, is_best, filename='checkpoint.pth.tar'):
    path_ = os.path.join("checkpoint_experts", "wts_100", "exp_21", "top-2")
    filepath = os.path.join(path_, filename)
    print (filepath)
    #torch.save(model_weights, filepath)
    if is_best:
        torch.save(model_weights, filepath)
        print ("******* Saved New PTH *********")

def to_csv(plots, name):  
    data_ = pd.DataFrame.from_dict(plots)
    data_.to_csv(name, index=False)



# This part of CAM script was taken from  https://github.com/chaeyoung-lee/pytorch-CAM
# I justed merge with my project.

# generate class activation mapping for the top1 prediction

'''
   TODOS:
     Merge following function with the stable_msnet.py script
     Perform CAM analysis for image mistake by router and corrected
     by the expert network.
'''

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (32, 32)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam(net, features_blobs, img_pil, classes, root_img):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    print (probs, idx)
    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(root_img)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.8 + img * 0.5
    cv2.imwrite('F:/Research/PHD_AIZU/tiny_ai/gradcam/pytorch-CAM/cam.jpg', result)
