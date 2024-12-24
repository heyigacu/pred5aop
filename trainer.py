
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim

from sklearn.metrics import mean_squared_error
import matplotlib
import copy
import joblib

from early_stop import EarlyStopping
from utils import *
from copy import deepcopy

PWD = os.path.abspath(os.path.dirname(__file__))


class train_mlp(object):
    ##############
    ## Binary-classify
    ##############    
    @staticmethod
    def train_multi_classify_kfolds(orimodel, kfolds=None, max_epochs=500, patience=10, save_folder=PWD+'/pretrained/', save_name='mlp_bi.pth', lr=0.001, weight_decay=0):
        val_metrics = []
        best_epochs = []
        preds_all = []
        labels_all = []
        for train_loader,val_loader in kfolds:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = "cpu"
            model = deepcopy(orimodel)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            early_stopping = EarlyStopping(save_folder,save_name,patience=patience)
            for epoch in range(1, max_epochs+1):
                model.train()
                for batch_idx,(train_features,train_labels) in enumerate(train_loader):
                    features, labels = train_features.to(device), train_labels.to(device)
                    preds = model(features)
                    loss = CrossEntropyLoss()(preds, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    for val_features, val_labels in val_loader:
                        features, labels = val_features.to(device), val_labels.to(device)
                        preds = model(features)
                        loss = CrossEntropyLoss()(preds, labels)
                        loss_val = loss.detach().item()
                        tmp_pred = preds.detach().cpu().numpy()
                        tmp_label = labels.cpu().numpy()
                        metrics_val = multi_classify_metrics(tmp_label, tmp_pred)
                early_stopping(epoch, loss_val, model, metrics_val, tmp_label, tmp_pred)
                if early_stopping.early_stop:
                    val_metrics.append(early_stopping.best_metrics)
                    best_epochs.append(early_stopping.best_epoch)
                    for arr in early_stopping.best_pred:
                        preds_all.append(list(arr))
                    labels_all+=list(early_stopping.best_label)
                    print("Early stopping")
                    break
            # print(np.array(labels_all),np.array(preds_all))
        multi_classify_metrics(np.array(labels_all), np.array(preds_all), plot_auc_curve=True,  plot_confuse_matrix=True, savename='multi', classnames=['NON-AOP','AOP'])

        return np.array(val_metrics).mean(0), np.array(best_epochs).mean(0)
    @staticmethod
    def train_bi_classify_all(model, all=None, epochs=500, save_folder=PWD+'/pretrained/',save_name='mlp_multi.pth', lr=0.001, weight_decay=0):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = "cpu"
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in range(1, epochs+1):
            loss_train = 0.
            acc_train = 0.
            model.train()
            for batch_idx,(train_features,train_labels) in enumerate(all):
                features, labels = train_features.to(device), train_labels.to(device)
                logits = model(features)
                loss = CrossEntropyLoss()(logits, labels)
                loss_train += loss.detach().item()
                acc = my_acc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                acc_train += acc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx+1)
            acc_train /= (batch_idx+1)
            torch.save(model.state_dict(), save_folder+save_name)
            if epoch%1 == 0:
                print('loss:',loss_train,'ACC:',acc_train)    
    @staticmethod
    def test_bi_classify(model, test=None, plot_cm=True, save_path=PWD+'/pretrained/mlp_multi.pth', classnames=['Coffee','Non-Coffee']):
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        model.eval()
        for i in list(test):
            features,labels = i
        preds = model(features)
        preds = preds.detach().cpu().numpy()
        rst = multi_classify_metrics(labels, preds, plot_cm=plot_cm, save_path=save_path, classnames=classnames)
        return rst
    