# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:23:11 2021

@author: Filippos
"""

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import os 
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from glob import glob
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec

dat = pd.read_csv('C:/Users/Filippos/Desktop/Master/data_subjects_info.csv')
dat.head()
dat.describe()

p = dat.hist(figsize = (20,20))

plt.matshow(dat.corr())
plt.colorbar()
plt.show()

sns.countplot(x=dat['gender'])
fig=plt.gcf()
fig.set_size_inches(6,4)






dat['age'].value_counts().plot(kind='bar', title='Age',figsize=(20,8)) 

dat['weight'].value_counts().plot(kind='bar', title='Weight',figsize=(20,8)) 

sns.regplot(x=dat['weight'], y=dat['height'])

# read many files csv 
import os
print(os.listdir("C:/Users/Filippos/Downloads/A_DeviceMotion_data/A_DeviceMotion_data"))

folders = glob('C:/Users/Filippos/Downloads/A_DeviceMotion_data/A_DeviceMotion_data/*_*')
folders = [s for s in folders if "csv" not in s]
df_all_list = []
activity_codes = {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
activity_types = list(activity_codes.keys())

for j in folders:
    print('j',j)
    csv = glob(j + '/*')
    for i in csv:
        df = pd.read_csv(i)
        df['activity'] = activity_codes[j[68:71]]
        df['sub_num'] = i[len(j)+5:-4]
       
        df_all_list.append(df)
        
        
df_all = pd.concat(df_all_list,axis=0)
df_all = df_all.drop('Unnamed: 0',axis=1)
print(df_all.shape)
print(df_all.columns)



fig, axs = plt.subplots(6,sharex=True, sharey=True)

fig.suptitle('Vertically stacked subplots')
activitycolors={0:'red',1:'orange',2:'green',3:'blue',4:'red',5:'grey'}
#Visualization
for act in activity_types: 
    axs[activity_codes[act]].set_title(activity_types[activity_codes[act]])
    df = df_all[(df_all['sub_num']=='3') & (df_all['activity']==activity_codes[act])]
    str1= 'tab:' 
    str2 =  activitycolors[activity_codes[act]] 
    vcolor =  str1+str2 
    axs[activity_codes[act]].plot(df_all['userAcceleration.x'][:500],vcolor)
