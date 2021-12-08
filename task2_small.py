#!/usr/bin/env python
# coding: utf-8

# In[101]:


from numpy.random import seed
import csv
import sqlite3
import time
import numpy as np
import random
import pandas as pd
from pandas import DataFrame
import scipy.sparse as sp
import math
import copy

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import KernelPCA

import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytorchtools import EarlyStopping
from pytorchtools import BalancedDataParallel
from radam import RAdam
import torch.nn.functional as F



import warnings
warnings.filterwarnings("ignore")

import os



# In[102]:


seed=0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


# In[103]:


def prepare(df_drug, feature_list,mechanism,action,drugA,drugB):

    d_label = {}
    d_feature = {}

    # Transfrom the interaction event to number
    d_event=[]
    for i in range(len(mechanism)):
        d_event.append(mechanism[i]+" "+action[i])

    count={}
    for i in d_event:
        if i in count:
            count[i]+=1
        else:
            count[i]=1
    event_num=len(count)
    list1 = sorted(count.items(), key=lambda x: x[1],reverse=True)
    each_event_num=[]
    for i in range(len(list1)):
        d_label[list1[i][0]]=i
        each_event_num.append(list1[i][1])

    vector = np.zeros((len(np.array(df_drug['name']).tolist()), 0), dtype=float)  #vector=[]
    for i in feature_list:
        #vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))#1258*1258
        tempvec=feature_vector(i, df_drug)
        vector = np.hstack((vector,tempvec))
    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['name']).tolist())):
        d_feature[np.array(df_drug['name']).tolist()[i]] = vector[i]

    # Use the dictionary to obtain feature vector and label
    new_feature = []
    new_label = []

    for i in range(len(d_event)):
        temp=np.hstack((d_feature[drugA[i]],d_feature[drugB[i]]))
        new_feature.append(temp)
        new_label.append(d_label[d_event[i]])

    new_feature = np.array(new_feature) #323539*....
    new_label = np.array(new_label)  #323539

    return new_feature, new_label, drugA,drugB,event_num,each_event_num


# In[104]:


def feature_vector(feature_name, df):
    def Jaccard(matrix):
        matrix = np.mat(matrix)

        numerator = matrix * matrix.T

        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T

        return numerator / denominator
    
    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1
    
    df_feature = np.array(df_feature)
    sim_matrix = np.array(Jaccard(df_feature))
    
    print(feature_name+" len is:"+ str(len(sim_matrix[0])))
    return sim_matrix


# In[105]:


class DDIDataset(Dataset):
    def __init__(self,x,y):
        self.len=len(x)
        self.x_data=torch.from_numpy(x)
        self.y_data=torch.from_numpy(y)
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len


# In[106]:


class MultiHeadAttention(torch.nn.Module):
    def __init__(self,input_dim,n_heads,ouput_dim=None):
        
        super(MultiHeadAttention, self).__init__()
        self.d_k=self.d_v=input_dim//n_heads
        self.n_heads = n_heads
        if ouput_dim==None:
            self.ouput_dim=input_dim
        else:
            self.ouput_dim=ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)
    def forward(self,X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q=self.W_Q(X).view( -1, self.n_heads, self.d_k).transpose(0,1)
        K=self.W_K(X).view( -1, self.n_heads, self.d_k).transpose(0,1)
        V=self.W_V(X).view( -1, self.n_heads, self.d_v).transpose(0,1)
        
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output


# In[107]:


class EncoderLayer(torch.nn.Module):
    def __init__(self,input_dim,n_heads):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(input_dim,n_heads)
        self.AN1=torch.nn.LayerNorm(input_dim)
        
        self.l1=torch.nn.Linear(input_dim, input_dim)
        self.AN2=torch.nn.LayerNorm(input_dim)
    def forward (self,X):
        
        output=self.attn(X)
        X=self.AN1(output+X)
        
        output=self.l1(X)
        X=self.AN2(output+X)
        
        return X


# In[108]:


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# In[109]:




# In[112]:

class feature_encoder(torch.nn.Module):  # twin network
    def __init__(self, vector_size,n_heads,n_layers):
        super(feature_encoder, self).__init__()

        self.layers = torch.nn.ModuleList([EncoderLayer(vector_size, n_heads) for _ in range(n_layers)])
        self.AN = torch.nn.LayerNorm(vector_size)

        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)

        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)

        self.l3 = torch.nn.Linear(vector_size // 4, vector_size//2)
        self.bn3 = torch.nn.BatchNorm1d(vector_size // 2)

        self.l4 = torch.nn.Linear(vector_size // 2, vector_size )


        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):

        for layer in self.layers:
            X = layer(X)
        X1=self.AN(X)
        X2 = self.dr(self.bn1(self.ac(self.l1(X1))))
        X3 = self.l2(X2)

        X4 = self.dr(self.bn3(self.ac(self.l3(self.ac(X3)))))
        X5 = self.l4(X4)

        return X1,X2,X3,X5
class feature_encoder2(torch.nn.Module):  # twin network
    def __init__(self, vector_size):
        super(feature_encoder2, self).__init__()

        self.l1 = torch.nn.Linear(vector_size, vector_size // 2)
        self.bn1 = torch.nn.BatchNorm1d(vector_size // 2)

        self.l2 = torch.nn.Linear(vector_size // 2, vector_size // 4)
        self.bn2 = torch.nn.BatchNorm1d(vector_size // 4)

        self.dr = torch.nn.Dropout(drop_out_rating)

        self.ac = gelu

    def forward(self, X):

        X = self.dr(self.bn1(self.ac(self.l1(X))))

        X = self.dr(self.bn2(self.ac(self.l2(X))))

        return X
class Model(torch.nn.Module):
    def __init__(self,input_dim,n_heads,n_layers,event_num):
        super(Model, self).__init__()

        self.input_dim = input_dim
        self.drugEncoder_input_dim=self.input_dim//2

        self.drugEncoderA=feature_encoder(self.drugEncoder_input_dim,n_heads,n_layers)
        self.drugEncoderB = feature_encoder(self.drugEncoder_input_dim, n_heads, n_layers)

        self.feaEncoder1_3_input_dim=self.drugEncoder_input_dim+self.drugEncoder_input_dim//4
        self.feaEncoder2_input_dim = self.drugEncoder_input_dim//2 + self.drugEncoder_input_dim// 2

        self.feaEncoder1 = feature_encoder2(self.feaEncoder1_3_input_dim)
        self.feaEncoder2 = feature_encoder2(self.feaEncoder2_input_dim)
        self.feaEncoder3 = feature_encoder2(self.feaEncoder1_3_input_dim)

        self.feaEncoder1_3_output_dim = self.feaEncoder1_3_input_dim//4
        self.feaEncoder2_output_dim = self.feaEncoder2_input_dim//4

        self.feaFui_input_dim = self.feaEncoder1_3_output_dim*2+self.feaEncoder2_output_dim+self.drugEncoder_input_dim//4*2

        self.feaFui = feature_encoder(self.feaFui_input_dim, n_heads, n_layers)

        self.linear_input_dim = self.feaFui_input_dim//4+self.feaFui_input_dim

        self.l1=torch.nn.Linear(self.linear_input_dim,(self.linear_input_dim+event_num)//2)
        self.bn1=torch.nn.BatchNorm1d((self.linear_input_dim+event_num)//2)

        self.l2 = torch.nn.Linear((self.linear_input_dim+event_num)//2, event_num)
        
        self.ac=gelu

        self.dr = torch.nn.Dropout(drop_out_rating)

        
    def forward(self, X):
        XA = X[:, 0:self.input_dim//2]
        XB = X[:, self.input_dim//2:]

        XA1,XA2,XA3,XAC=self.drugEncoderA(XA)
        XB1, XB2, XB3 ,XBC= self.drugEncoderB(XB)

        XDC = torch.cat((XAC, XBC), 1)

        X1 = torch.cat((XA1,XB3), 1)
        X2 = torch.cat((XA2, XB2), 1)
        X3 = torch.cat((XA3, XB1), 1)

        X1=self.feaEncoder1(X1)
        X2 = self.feaEncoder2(X2)
        X3 = self.feaEncoder3(X3)

        XC = torch.cat((X1, X2, X3,XA3,XB3), 1)
        _,_,XC,_=self.feaFui(XC)

        X = torch.cat((XA3,XB3,X1, X2, X3,XC), 1)

        X=self.dr(self.bn1(self.ac(self.l1(X))))

        X=self.l2(X)
        
        return X,XC,XDC


# In[114]:
class LabelSmoothing(nn.Module):
    "Implement label smoothing.  size=class number  "

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(size_average=False)

        self.confidence = 1.0 - smoothing

        self.smoothing = smoothing

        self.size = size

        self.true_dist = None

        self.Logsoftmax = nn.LogSoftmax()

    def forward(self, x, target):

        x = self.Logsoftmax(x)

        assert x.size(1) == self.size
        true_dist = x.data.clone()

        true_dist.fill_(self.smoothing / (self.size - 1))

        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        self.true_dist = true_dist

        return self.criterion(x, torch.autograd.Variable(true_dist, requires_grad=False).to(device))

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):

        labels = labels.view(-1, 1) # [B * S, 1]
        preds = preds.view(-1, preds.size(-1)) # [B * S, C]
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, labels)
        preds_logsoft = preds_logsoft.gather(1, labels)
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class con_loss(nn.Module):
    def __init__(self, T=0.05):
        super(con_loss, self).__init__()

        self.T = T

    def forward(self, representations, label):
        n = label.shape[0]
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
        mask_no_sim = torch.ones_like(mask) - mask

        mask_dui_jiao_0 = (torch.ones(n, n) - torch.eye(n, n)).to(device)

        similarity_matrix = torch.exp(similarity_matrix / self.T)

        similarity_matrix = similarity_matrix * mask_dui_jiao_0

        sim = mask * similarity_matrix

        no_sim = similarity_matrix - sim

        no_sim_sum = torch.sum(no_sim, dim=1)

        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum = sim + no_sim_sum_expend
        loss = torch.div(sim, sim_sum)
        loss = mask_no_sim + loss + torch.eye(n, n).to(device)
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)

        return loss
class my_loss1(nn.Module):
    def __init__(self,classNum):
        
        super(my_loss1,self).__init__()
        
        self.criteria1 = LabelSmoothing(classNum, smoothing=label_smoothing)
        self.criteria2=torch.nn.MSELoss()
        self.criteria3 = con_loss(T=con_loss_T)

    def forward(self, X, target,XC,XDC,inputs):

        loss=self.criteria1(X,target)+ \
             10*self.criteria2(inputs.float(), XDC)+\
             0.1*self.criteria3(XC,target)

        return loss
class my_loss2(nn.Module):
    def __init__(self,classNum,each_num_wei):
        
        super(my_loss2,self).__init__()
        
        self.criteria1 = focal_loss(alpha=each_num_wei,num_classes=classNum)
        self.criteria2=torch.nn.MSELoss()
        self.criteria3 = con_loss(T=con_loss_T)

    def forward(self, X, target,XC,XDC,inputs):
        loss = self.criteria1(X, target) + \
               10*self.criteria2(inputs.float(), XDC) + \
               0.1*self.criteria3(XC, target)
        return loss


def BERT_train(model,x_train,y_train,x_test,y_test,event_num,each_num_wei):

    model_optimizer=RAdam(model.parameters(),lr=learn_rating,weight_decay=weight_decay_rate)
    model=torch.nn.DataParallel(model)
    model=model.to(device)

    x_train=np.vstack((x_train,np.hstack((x_train[:,len(x_train[0])//2:],x_train[:,:len(x_train[0])//2]))))
    y_train = np.hstack((y_train, y_train))
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    len_train=len(y_train)
    len_test=len(y_test)
    print("arg train len", len(y_train))
    print("test len", len(y_test))

    train_dataset = DDIDataset(x_train,np.array(y_train))
    test_dataset = DDIDataset(x_test,np.array(y_test))
    train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    for epoch in range(epo_num):
        if epoch<epoch_changeloss:
            my_loss=my_loss1(event_num)
        else:
            my_loss=my_loss2(event_num,each_num_wei)

        running_loss = 0.0

        model.train()
        for batch_idx,data in enumerate(train_loader,0):
            inputs, targets = data

            inputs=inputs.to(device)
            targets=targets.to(device)

            model_optimizer.zero_grad()

            X,XC,XDC=model(inputs.float())

            loss= my_loss(X, targets,XC,XDC,inputs)

            loss.backward()
            model_optimizer.step()   
            running_loss += loss.item()

        model.eval()
        testing_loss=0.0
        with torch.no_grad():
            for batch_idx,data in enumerate(test_loader,0):
                inputs,target=data

                inputs = inputs.to(device)

                target=target.to(device)

                X,XC,XDC=model(inputs.float())

                loss=my_loss(X, target, XC,XDC,inputs)
                testing_loss += loss.item()
        print('epoch [%d] loss: %.6f testing_loss: %.6f ' % (epoch+1,running_loss/len_train,testing_loss/len_test))

    pre_score=np.zeros((0, event_num), dtype=float)
    model.eval()        
    with torch.no_grad():
        for batch_idx,data in enumerate(test_loader,0):
            inputs,_=data
            inputs=inputs.to(device)
            X, _,_= model(inputs.float())
            pre_score =np.vstack((pre_score,F.softmax(X).cpu().numpy()))
    return pre_score


# In[116]:

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)
    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 6
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, np.arange(event_num))

    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    # result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[2] = roc_auc_score(y_one_hot, pred_score, average='micro')
    # result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    # result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[3] = f1_score(y_test, pred_type, average='macro')
    # result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[4] = precision_score(y_test, pred_type, average='macro')
    # result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[5] = recall_score(y_test, pred_type, average='macro')

    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        # result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),average=None)
        result_eve[i, 2] = 0.0
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]


# In[117]:


def cross_val(feature,label,drugA,drugB,event_num,each_event_num):

    y_true = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    y_pred = np.array([])

    # cro val
    temp_drug1 = [[] for i in range(event_num)]
    temp_drug2 = [[] for i in range(event_num)]
    for i in range(len(label)):
        temp_drug1[label[i]].append(drugA[i])
        temp_drug2[label[i]].append(drugB[i])
    drug_cro_dict = {}
    for i in range(event_num):
        for j in range(len(temp_drug1[i])):
            drug_cro_dict[temp_drug1[i][j]] = j % cross_ver_tim
            drug_cro_dict[temp_drug2[i][j]] = j % cross_ver_tim
    train_drug = [[] for i in range(cross_ver_tim)]
    test_drug = [[] for i in range(cross_ver_tim)]
    for i in range(cross_ver_tim):
        for dr_key in drug_cro_dict.keys():
            if drug_cro_dict[dr_key] == i:
                test_drug[i].append(dr_key)
            else:
                train_drug[i].append(dr_key)

    each_event_num_sum = sum(each_event_num)
    each_num_wei_dao = [each_event_num_sum / x for x in each_event_num]
    each_num_wei_dao_sum = sum(each_num_wei_dao)
    each_num_wei = [50*math.log(x / each_num_wei_dao_sum * 1000000) for x in each_num_wei_dao]
    print(each_num_wei)

    for cross_ver in range(cross_ver_tim):
        
        model=Model(len(feature[0]),bert_n_heads,bert_n_layers,event_num)

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        for i in range(len(drugA)):
            if (drugA[i] in np.array(train_drug[cross_ver])) and (drugB[i] in np.array(train_drug[cross_ver])):
                X_train.append(feature[i])
                y_train.append(label[i])

            if (drugA[i] not in np.array(train_drug[cross_ver])) and (drugB[i] in np.array(train_drug[cross_ver])):
                X_test.append(feature[i])
                y_test.append(label[i])

            if (drugA[i] in np.array(train_drug[cross_ver])) and (drugB[i] not in np.array(train_drug[cross_ver])):
                X_test.append(feature[i])
                y_test.append(label[i])


        print("train len", len(y_train))
        print("test len", len(y_test))

        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        
        pred_score=BERT_train(model,X_train,y_train,X_test,y_test,event_num,each_num_wei)
        
        pred_type = np.argmax(pred_score, axis=1)
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.row_stack((y_score, pred_score))

        y_true = np.hstack((y_true, y_test))
        
    result_all, result_eve= evaluate(y_pred, y_score, y_true, event_num)

    return result_all, result_eve



# In[118]:






# In[119]:


def main():
    

    mechanism = extraction['mechanism']
    action = extraction['action']
    drugA = extraction['drugA']
    drugB = extraction['drugB']
    

    
    new_feature, new_label, drugA,drugB,event_num,each_event_num=prepare(df_drug,feature_list,mechanism,action,drugA,drugB)
    np.random.seed(seed)
    np.random.shuffle(new_feature)
    np.random.seed(seed)
    np.random.shuffle(new_label)
    np.random.seed(seed)
    np.random.shuffle(drugA)
    np.random.seed(seed)
    np.random.shuffle(drugB)
    print("dataset len", len(new_feature))
    
    start=time.time()
    result_all, result_eve=cross_val(new_feature,new_label,drugA,drugB,event_num,each_event_num)
    print("time used:", (time.time() - start) / 3600)
    save_result(file_path,"all",result_all)
    save_result(file_path,"each",result_eve)


# In[120]:
def save_result(filepath,result_type,result):
    with open(filepath+result_type +'task2_small'+ '.csv', "w", newline='',encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

file_path="/home/dqw_wyj/conLea/"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conn = sqlite3.connect("/home/dqw_wyj/DDIMDL/event.db")
df_drug = pd.read_sql('select * from drug;', conn)
extraction = pd.read_sql('select * from extraction;', conn)

label_smoothing=0.3
con_loss_T=0.05
learn_rating=0.00002
batch_size=512
epo_num=120
epoch_changeloss=epo_num//3

bert_n_heads=4
bert_n_layers=2
drop_out_rating=0.5
cross_ver_tim=5
weight_decay_rate=0.0001
feature_list = ["smile","target","enzyme","pathway"]

main()

