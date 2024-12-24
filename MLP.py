
import os
import pandas as pd
import numpy as np
import venn
from collections import Counter

import matplotlib 
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

from data_clean import data_preprocess
from load_data import *
from model import MLP
from trainer import train_mlp

parent_dir = os.path.abspath(os.path.dirname(__file__))
print(parent_dir)

def bi_classify():
    data_path = parent_dir+"/dataset/train/cleaned_data_bi.csv"
    df = pd.read_csv(data_path,sep='\t',header=0)
    sequences = df['Sequence']
    labels = list(df['Label'].astype(int))
    print(Counter(labels))
    nBits = 2048
    features = np.array([morgan_featurizer(sequence, nBits=nBits) for sequence in sequences])
    # features = np.array([esm_featurizer(sequence) for sequence in sequences])
    tuple_ls = list(zip(features, labels))

    
    batchsize = 32
    drop_last = True
    kfolds = load_data_kfold_batchsize(tuple_ls, batchsize=batchsize, Stratify=True, drop_last=drop_last, sample_method=SMOTEOS)

    n_hiddens = 256
    n_tasks = 2
    model = MLP(n_feats=nBits, n_hiddens=n_hiddens, n_tasks=n_tasks)
    print('mlp start training!')
    rst_mlp, best_epoch = train_mlp.train_multi_classify_kfolds(model, kfolds=kfolds, max_epochs=500, patience=7, save_folder=parent_dir+'/pretrained/',save_name='kfolds.pth')
    print('optimization finished!', rst_mlp)
    print(best_epoch)
    print(rst_mlp)

    # model = MLP(n_feats=nBits, n_hiddens=n_hiddens, n_tasks=n_tasks)
    # all = load_data_all_batchsize(tuple_ls, batchsize, drop_last=drop_last)
    # train_mlp.train_bi_classify_all(model, all=all, epochs=int(best_epoch)+1, save_folder=parent_dir+'/pretrained/',save_name='all_mlp_bi.pth')


def multi_classify(nBits=2048, batchsize=32, n_hiddens=256, lr=0.001, active_func=None):
    data_path = parent_dir+"/dataset/train/cleaned_data_multi.csv"
    df = pd.read_csv(data_path,sep='\t',header=0)
    sequences = df['Sequence']
    labels = list(df['Label'].astype(int))
    print(Counter(labels))
    nBits = nBits
    features = np.array([morgan_featurizer(sequence, nBits=nBits) for sequence in sequences])
    features, labels = SMOTEOS(features, labels)
    # features = np.array([esm_featurizer(sequence) for sequence in sequences])
    tuple_ls = list(zip(features, labels))

    batchsize = batchsize
    drop_last = True
    kfolds = load_data_kfold_batchsize(tuple_ls, batchsize=batchsize, Stratify=True, drop_last=drop_last, sample_method=None)

    n_hiddens = n_hiddens
    n_tasks = 5
    model = MLP(n_feats=nBits, n_hiddens=n_hiddens, n_tasks=n_tasks, active_func=active_func)
    print('mlp start training!')
    rst_mlp, best_epoch = train_mlp.train_multi_classify_kfolds(model, kfolds=kfolds, max_epochs=500, patience=7, lr=lr, save_folder=parent_dir+'/pretrained/',save_name='kfolds.pth')
    print('optimization finished!', rst_mlp)
    print(best_epoch)
    print(rst_mlp)
    return rst_mlp
    # model = MLP(n_feats=nBits, n_hiddens=n_hiddens, n_tasks=n_tasks)
    # all = load_data_all_batchsize(tuple_ls, batchsize, drop_last=drop_last)
    # train_mlp.train_bi_classify_all(model, all=all, epochs=int(best_epoch)+1, save_folder=parent_dir+'/pretrained/',save_name='all_mlp_multi.pth')

if __name__ == '__main__':
    bi_classify()
    # dic = {}
    # dic['nBits'] = []
    # dic['n_hiddens'] = []
    # dic['lr'] = []
    # dic['active_func'] = []
    # ls = []
    # for nBits in [512, 1024, 2048]:
    #     for n_hiddens in [128, 256, 512, 1024]:
    #         for lr in [0.0001, 0.0005, 0.001]:
    #             for active_func in ['ReLU', 'Sigmoid', 'GELU']:
    #                 rst_mlp = multi_classify(nBits, 64, n_hiddens, lr, active_func)
    #                 dic['nBits'].append(nBits)
    #                 dic['n_hiddens'].append(n_hiddens)
    #                 dic['lr'].append(lr)
    #                 dic['active_func'].append(active_func)
    #                 ls.append(rst_mlp)
    # df1 = pd.DataFrame(dic)
    # df2 = pd.DataFrame(ls, columns=None)
    # df2.columns = ['pre','rec','acc','f1','mcc','auc']
    # df = pd.concat([df1,df2], axis=1)
    # df.to_csv('adj_para_rst.csv',sep='\t')
    
    
