import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import os

work_dir = os.path.abspath(os.path.dirname(__file__))

def validate(df):
    sequences = df['Sequence']
    labels = df['Label']
    vect=CountVectorizer(analyzer='char_wb',ngram_range=(3,3))
    vect.fit(sequences)
    kf = StratifiedKFold(n_splits=5,shuffle=True)
    for train_idxs,val_idxs in kf.split(sequences, labels):
        train_features = vect.transform([sequences[index] for index in train_idxs])
        train_labels = [labels[index] for index in train_idxs]
        test_features = vect.transform([sequences[index] for index in val_idxs])
        test_labels = [labels[index] for index in val_idxs]
        model=MultinomialNB()
        model.fit(train_features,train_labels)
        NB_pred=model.predict(test_features)
        score = accuracy_score(test_labels, NB_pred)
        print(score)

def predit(df_train, df_predict, threshold):
    # featurize
    labels = list(df_train['Label'])
    sequences_predit = list(df_predict['Sequence'])
    sequences_train = list(df_train['Sequence'])

    sequences_total = sequences_train + sequences_predit
    vect=CountVectorizer(analyzer='char_wb',ngram_range=(3,3))
    vect.fit(sequences_total)

    features_train = vect.transform(sequences_train)
    features_predit = vect.transform(sequences_predit)

    # train
    model=MultinomialNB()
    model.fit(features_train,labels)

    # predict
    NB_pred=model.predict_proba(features_predit)
    # NB_pred=model.predict(features_predit)
    df_rst = pd.DataFrame(NB_pred)
    df_rst.columns =  ['AOP','AMP','ACP', 'NP','ACEIP']
    df_rst.insert(0, 'Sequence', sequences_predit)
    df_rst = df_rst[df_rst['AOP'] > threshold]
    df_rst.to_csv(work_dir+'/aop_word.csv',index=True,header=True,sep='\t')
    return df_rst

if __name__ == '__main__':
    df_train = pd.read_csv(work_dir+'/dataset/train/cleaned_data_multi.csv',sep='\t',header=0)
    validate(df_train)
    df_predict = pd.read_csv(work_dir+'/dataset/rpg/34_ep.csv',sep='\t',header=0)
    predit(df_train,df_predict,0.9)
