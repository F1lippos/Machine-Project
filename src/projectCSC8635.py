# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:23:11 2021

@author: Filippos
"""
######################################
#Libraries Import
######################################
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


##################################
# Data Import
##################################
dat = pd.read_csv('C:/Users/Filippos/Desktop/Master/data_subjects_info.csv')

##################################
# Data Overview 
##################################
dat.head()
dat.info()
dat.describe()
dat.isna().sum()

#################################
# Data Analysis using Panda
#################################

# to see the corelattion of the columns
dat.corr()

corr_matrix = dat.corr()
corr_matrix["age"].sort_values(ascending=False)
corr_matrix["height"].sort_values(ascending=False)

dat.groupby(["gender"]).count()
dat.groupby(["gender"]).size()


################################
# Data VIsualization
################################

# Plot 1
p = dat.hist(figsize = (20,20))

plt.matshow(dat.corr())
plt.colorbar()
plt.show()


# Plot 2
sns.pairplot(dat, size=3)
sns.pairplot(dat,hue ='age', size=3)

# Plot 3
sns.countplot(x=dat['gender'])
fig=plt.gcf()
fig.set_size_inches(6,4)


# Plot 4
sns.regplot(x=dat['weight'], y=dat['height'])

dat['age'].value_counts().plot(kind='bar', title='Age',figsize=(20,8)) 

dat['weight'].value_counts().plot(kind='bar', title='Weight',figsize=(20,8)) 

sns.regplot(x=dat['weight'], y=dat['height'])


#plot 5
dat[['weight','height','age']].boxplot()
fig = plt.gcf()
fig.set_size_inches(15, 10)

##################################
# Data Import many files csv 
##################################
 
import os
print(os.listdir("C:/Users/Filippos/Downloads/A_DeviceMotion_data/A_DeviceMotion_data"))

folders = glob('C:/Users/Filippos/Downloads/A_DeviceMotion_data/A_DeviceMotion_data/*_*')
folders = [s for s in folders if "csv" not in s]
df_all_list = []
df_sample = []
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
        df_sample.append(df.sample(n=300))
        
        
df_all = pd.concat(df_all_list,axis=0)
df_all = df_all.drop('Unnamed: 0',axis=1)
df_sample_all = pd.concat(df_sample,axis =0)
df_sample_all = df_sample_all.drop('Unnamed: 0',axis=1)

##################################
# Data Overview 
##################################
df_all.head()
df_all.info()
df_all.describe()
df_all.isna().sum()

df_sample_all.head()
df_sample_all.info()
df_sample_all.describe()
df_sample_all.isna().sum()

df_sample_all.groupby(["activity"]).size()
df_sample_all.groupby(["activity","sub_num"]).size()



print(df_all.shape)
print(df_all.columns)

#################################
# Data Analysis using Panda of csv files 
#################################

# to see the corelattion of the columns
df_sample_all.corr()


################################
# Data VIsualization
################################
#plot1 
p = df_all.hist(figsize = (20,20))

plt.matshow(dat.corr())
plt.colorbar()
plt.show()


# plot 2 

fig, axs = plt.subplots(6,sharex=True, sharey=True)

fig.suptitle('Vertically stacked subplots')
activitycolors={0:'red',1:'orange',2:'green',3:'blue',4:'red',5:'grey'}
#Visualization
for act in activity_types: 
    axs[activity_codes[act]].set_title(activity_types[activity_codes[act]])
    dfp = df_all[(df_all['sub_num']=='3') & (df_all['activity']==activity_codes[act])]
    str1= 'tab:' 
    str2 =  activitycolors[activity_codes[act]] 
    vcolor =  str1+str2 
    axs[activity_codes[act]].plot(dfp['userAcceleration.x'][:500],vcolor)



#Value Distribution of the recorded Sensor Data
df_all[['userAcceleration.x','userAcceleration.y','userAcceleration.z']].describe()  


df_all.columns[2:-2]


df_all[['userAcceleration.x','userAcceleration.y','userAcceleration.z']].boxplot()
fig = plt.gcf()
fig.set_size_inches(15, 10)




### Length of time series
series_length = list()
for act in activity_types:
    print(act)
    for sub in range(1,25):
        sub = str(sub)
        print(sub)
        series_length.append(df_all[(df_all['sub_num']==sub) & (df_all['activity']==activity_codes[act])].shape[0])
plt.figure(2)
plt.title('Histogram of length of raw time series')
plt.hist(series_length,rwidth=0.5,align='left')


# Plot class distribution:
activiry_counts = df_all.activity.apply(lambda x: activity_types[x] ).value_counts()
activiry_counts.plot(kind='bar', title='Activity Class Distibution')
plt.show()


#############################################
##  MODELING 
#############################################
dat.dtypes
df_all.dtypes

dfML = df_all[["userAcceleration.x","activity","sub_num"]]

dfML.shape
dfML.dtypes
#dfML["subnum"] =  dfML["sub_num"].astype(str).astype(int)
dfML["subnum"] =  dfML["sub_num"].astype(str)
dat["code"] = dat["code"].astype(str)

pd.merge(dfML, dat, left_on='sub_num', right_on='code', how='inner').drop('code', axis=1)
dfPersons =  pd.merge(dfML, dat, left_on='sub_num', right_on='code', how='inner').drop('code', axis=1)



array = dfPersons.values
array.dtypes
X = array[:,1:4]
y = array[:,0]
 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


    