import pandas as pd
import numpy as np
import random
import os
from tqdm import tqdm

df = pd.read_csv('./data/train.csv')
vc = df['label_group'].value_counts()
vc = pd.DataFrame(vc)
vc['index'] = vc.index
vc = vc.sort_values(by=['label_group', 'index'], ascending=[False, True])
vc = vc['label_group']

less_than6 = []
more_than6 = []
for i in range(len(vc)):
    if vc.values[i] < 6:
        less_than6.append(vc.index[i])
    else:
        more_than6.append(vc.index[i])

train = pd.DataFrame(columns=df.columns)
valid = pd.DataFrame(columns=df.columns)
test = pd.DataFrame(columns=df.columns)

random.seed(609)
choices = np.array(random.sample(range(len(less_than6)), len(less_than6))) / len(less_than6)

print('start splitting')
for i in tqdm(range(len(less_than6))):
    name = less_than6[i:i+1]
    choice = choices[i]
    if choice < 0.6:
        train = train.append(df[df['label_group'].isin(name)])
    elif choice >= 0.6 and choice < 0.8:
        valid = valid.append(df[df['label_group'].isin(name)])
    else:
        test = test.append(df[df['label_group'].isin(name)])

for i in tqdm(range(len(more_than6))):
    name = more_than6[i:i+1]
    tmp_df = df[df['label_group'].isin(name)]
    length = len(tmp_df)
    if length < 10:
        num = [length-4, length-2]
    else:
        num = [int(0.6*length), int(0.8*length)]
    train = train.append(tmp_df.iloc[:num[0], :])
    valid = valid.append(tmp_df.iloc[num[0]:num[1], :])
    test = test.append(tmp_df.iloc[num[1]:, :])

assert len(train) + len(valid) + len(test) == len(df)

if not os.path.exists('./data/splitted'):
    os.mkdir('./data/splitted')
train.to_csv('./data/splitted/train.csv', header=df.columns, index=None)
valid.to_csv('./data/splitted/valid.csv', header=df.columns, index=None)
test.to_csv('./data/splitted/test.csv', header=df.columns, index=None)