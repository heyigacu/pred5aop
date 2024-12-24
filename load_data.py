

from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from imblearn.over_sampling import SMOTE
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
import pandas as pd
from esm_embedding.embedding import esm_embeddings
from collections import Counter

def SMOTEOS(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y) 
    return X_resampled, y_resampled


def load_data_all_batchsize(tuple_ls, batchsize, drop_last=True):
    """
    args:
        ls: [(feature,label)]
        batchsize: int
    """
    return DataLoader(tuple_ls, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)

def load_data_kfold_batchsize(tuple_ls, batchsize, Stratify=True, drop_last=True, sample_method=None):
    """
    args:
        ls: [(feature,label)]
        batchsize: int
    """
    features, labels = list(zip(*tuple_ls))
    if Stratify:
        kf = StratifiedKFold(n_splits=5,shuffle=True)
    else:
        kf = KFold(n_splits=5,shuffle=True)
    kfolds=[]
    for train_idxs,val_idxs in kf.split(features, labels):
        train_features = [features[index] for index in train_idxs]
        train_labels = [labels[index] for index in train_idxs]
        if sample_method != None:
            train_features, train_labels = sample_method(train_features, train_labels)
            print('afer sampling:',Counter(train_labels))
        trains = list(zip(np.array(train_features),train_labels))
        trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)
        vals = [tuple_ls[index] for index in val_idxs]
        vals = DataLoader(vals,batch_size=len(vals), shuffle=True,)
        kfolds.append((trains,vals))
    return kfolds

def morgan_featurizer(sequence, nBits=2048):
    mol = Chem.MolFromFASTA(sequence)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits))

def esm_featurizer(sequence):
    return np.array(esm_embeddings([('0',sequence)]))[0]

def multi_times_embeddings(fn, ls, batch):
    tasks = list(range(0,len(ls),batch))
    for i in range(len(tasks)):
        if i == 0:
            features = np.array(fn(ls[tasks[i]:tasks[i+1]]))
        elif i != (len(tasks)-1):
            features = np.concatenate((features,np.array(fn(ls[tasks[i]:tasks[i+1]]))),axis=0)
        else:
            features = np.concatenate((features,np.array(fn(ls[tasks[i]:]))),axis=0)
    return features

def esm_embedding_for_sequences(sequences, names=None, batch=None):
    name_sequence_tuple_ls = []
    for i,sequence in enumerate(sequences):
        name_sequence_tuple_ls.append((str(i), sequence))
    if batch==None:
        features = np.array(esm_embeddings(name_sequence_tuple_ls))
    else:
        features = multi_times_embeddings(esm_embeddings, name_sequence_tuple_ls, batch)
    return features

