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
