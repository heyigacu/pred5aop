import argparse
import torch
import os
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import pandas as pd
from model import MLP
from load_data import morgan_featurizer,esm_featurizer

work_dir = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='peptide bio-activity predictor')
parser.add_argument("-i", "--file", type=str, default=work_dir+'/dataset/rpg/34_ep.csv', help="input file")
parser.add_argument("-o", "--out", type=str, default=work_dir+'/result.csv',help="output file")
parser.add_argument("-m", "--model", type=str, choices=['multi','bi'],default='multi')
args = parser.parse_args()

##################
# predict
##################
nBits = 2048
nhiddens = 256
def mlp_multi(model, path, sequences):
    state_dict = torch.load(work_dir+"/"+path)
    model.load_state_dict(state_dict)
    model.eval()
    total = []
    for i,sequence in enumerate(sequences):
        if args.model == 'multi':
            try:
                if i % 10000 ==0:
                    print(i)
                for i in list(DataLoader([morgan_featurizer(sequence,nBits=nBits)], batch_size=1, shuffle=False, collate_fn=None, drop_last=False)):
                    feature = i
                rst = model(feature)
                rst =  F.softmax(rst,dim=1).detach().numpy()[0]
                labels = ['AOP','AMP','ACP', 'NP','ACEIP']
                string = labels[rst.argmax()]
                ls = []
                ls.append(string)
                for value in rst:
                    ls.append('{:.4f}'.format(value))
                total.append(ls)
            except:
                total.append(['error sequence', 'nan', 'nan', 'nan', 'nan', 'nan'])
        else:
            try:
                if i % 10000 ==0:
                    print(i)
                for i in list(DataLoader([morgan_featurizer(sequence,nBits=nBits)], batch_size=1, shuffle=False, collate_fn=None, drop_last=False)):
                    feature = i
                rst = model(feature)
                rst =  F.softmax(rst,dim=1).detach().numpy()[0]
                labels = ['Non-AOP','AOP']
                string = labels[rst.argmax()]
                ls = []
                ls.append(string)
                for value in rst:
                    ls.append('{:.4f}'.format(value))
                total.append(ls)
            except:
                total.append(['error sequence', 'nan', 'nan'])  
    return total

sequences = list(pd.read_csv(args.file, header=0, sep='\t')['Sequence'])

if args.model == 'multi':
    model = MLP(n_feats = nBits, n_hiddens = nhiddens, n_tasks = 5)
    path = "/pretrained/all_mlp_bio.pth"
    total = mlp_multi(model, path, sequences)
    df = pd.DataFrame(total)
    df.columns =  ['Bio-activity', 'AOP','AMP','ACP', 'NP','ACEIP']
else:
    model = MLP(n_feats = nBits, n_hiddens = nhiddens, n_tasks = 2)
    path = "/pretrained/all_mlp_old.pth"
    total = mlp_multi(model, path, sequences)
    df = pd.DataFrame(total)
    df.columns =  ['Bio-activity', 'Non-AOP', 'AOP']

df.insert(0,'Sequence',sequences)
df.to_csv(args.out,index=False,header=True,sep='\t')

    

