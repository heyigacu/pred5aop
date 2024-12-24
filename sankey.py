import pandas as pd 
import os
from collections import Counter
import plotly.graph_objects as go

def create_sankey_diagram(labels, sources, targets, values, title="Sankey Diagram"):
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=5,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    ))
    fig.update_layout(title_text=title, font_size=13)
    fig.show()



df = pd.read_csv('result.csv', sep='\t')
result_dict = dict(zip(df['Sequence'], df['Bio-activity']))

# print(result_dict)

def contains_invalid_chars(s):
    invalid_chars = {'B', 'X', 'U', 'J', 'O', 'Z'}
    return all(char not in invalid_chars for char in s)


labels = []
sources = []
targets = []
values = []
for file in os.listdir('dataset/rpg/ep_screen'):
    labels.append(file.split('.')[0])
labels += ['ACEIP', 'AOP', 'ACP', 'AMP', 'NP']

for file in os.listdir('dataset/rpg/ep_screen'):
    df = pd.read_csv('dataset/rpg/ep_screen/'+file, sep='\t', header=None)
    ls = list(df.iloc[:, 0])
    bios = []
    for seq in ls:
        if contains_invalid_chars(seq):
            bios.append(result_dict[seq])
    frequency = dict(Counter(bios))
    for k,v in frequency.items():
        sources.append(labels.index(file.split('.')[0]))
        targets.append(labels.index(k))
        values.append(v)


create_sankey_diagram(labels, sources, targets, values, title="Sankey Diagram: Species for peptides and Bio-Activity")

