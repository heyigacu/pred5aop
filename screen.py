import pandas as pd
import os
from MultinomialNB import predit

work_dir = os.path.abspath(os.path.dirname(__file__))
df = pd.read_csv(work_dir+'/result.csv',header=0,sep='\t')
print(df['Bio-activity'].value_counts())
df = df[df['AOP'] > 0.9999]
df.to_csv(work_dir+'/screen.csv',sep='\t',header=True,index=False)
