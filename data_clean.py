import pandas as pd
import numpy as np
import os
parent_dir = os.path.abspath(os.path.dirname(__file__))
print(parent_dir)

def data_preprocess():
    import rdkit
    import rdkit.Chem as Chem

    df = pd.read_csv(parent_dir+"/dataset/train/data.csv", header=0, sep="\t")
    print(df['Label'].value_counts())

    df['Sequence'] = [seq.upper() for seq in list(df['Sequence'])]

    # delete can't recognize by rdkit
    none_list=[]
    for index, row in df.iterrows():
        if Chem.MolFromFASTA(row['Sequence']) is None:
            none_list.append(index)
    print("dropping {} unidentified molecules".format(len(none_list)))
    df=df.drop(none_list)

    # delete number < 3
    none_list=[]
    for index, row in df.iterrows():
        if len(list(row['Sequence'])) < 3 or len(list(row['Sequence'])) > 20:
            none_list.append(index)
    print("dropping {} molecules with lenth < 3".format(len(none_list)))
    df=df.drop(none_list)
    print(df['Label'].value_counts())

    # drop duplicates
    df_new = pd.DataFrame()
    for key in df['Label'].value_counts().to_dict().keys():
        df_ = df.query('Label == @key')
        df_ = df_.drop_duplicates(subset=['Sequence'], keep='first')
        df_new = pd.concat([df_new,df_],axis = 0)
    for key in df_new['Label'].value_counts().to_dict().keys():
        df_ = df_new.query('Label == @key')
        print(len(list(set(list(df_['Sequence'])))),len(list(df_['Sequence'])))

    df_new.to_csv(parent_dir+"/dataset/train/cleaned_data_multi.csv", header=True, sep="\t", index=False)

    df_new.loc[df['Label'] > 1, 'Label'] = 0
    df_new.loc[df['Label'] < 2, 'Label'] = 1
    df_new = df_new.drop_duplicates(subset=['Sequence'], keep='first')
    df_new.to_csv(parent_dir+"/dataset/train/cleaned_data_bi.csv", header=True, sep="\t", index=False)

if __name__ == "__main__":
    data_preprocess()




