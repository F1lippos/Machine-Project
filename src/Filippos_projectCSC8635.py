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
dat = pd.read_csv('C:/Users/Filippos/Desktop/Master/CSC8635-Machine Project/data/data_subjects_info.csv')
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
##p = dat.hist(figsize = (20,20))

## Correlation
plt.figure(figsize=(10,10))
corr = dat.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, annot=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=45,
);

# Plot 2

temp = dat.copy()
temp=temp.drop(columns=['code'])
temp.dtypes
sns.pairplot(temp, size=3)

sns.pairplot(dat, size=3)

sns.pairplot(dat,hue ='age', size=3)

# Plot 3
sns.countplot(x=dat['gender'])
fig=plt.gcf()
fig.set_size_inches(6,4)


# Plot 4
sns.regplot(x=dat['weight'], y=dat['height'])

#dat['age'].value_counts().plot(kind='bar', title='Age',figsize=(20,8)) 

#dat['weight'].value_counts().plot(kind='bar', title='Weight',figsize=(20,8)) 

#sns.regplot(x=dat['weight'], y=dat['height'])


#plot 5
dat[['weight','height','age']].boxplot()
fig = plt.gcf()
fig.set_size_inches(15, 10)

##################################
# Data Import many files csv 
##################################
 
import os
print(os.listdir("C:/Users/Filippos/Desktop/Master/CSC8635-Machine Project/data/A_DeviceMotion_data/A_DeviceMotion_data"))
folders = glob('C:/Users/Filippos/Desktop/Master/CSC8635-Machine Project/data/A_DeviceMotion_data/A_DeviceMotion_data/*_*')


folders = [s for s in folders if "csv" not in s]
df_all_list = []
df_sample = []
activity_codes = {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
activity_types = list(activity_codes.keys())

for j in folders:
    ##print('j',j)
    csv = glob(j + '/*')
    for i in csv:
        #print('load file of participant',i)
        df = pd.read_csv(i)
        df['activity'] = activity_codes[j[102:105]]
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
#p = df_all.hist(figsize = (20,20))

#plt.matshow(df_sample_all.corr())
#plt.colorbar()
#plt.show()

## Corellation 
plt.figure(figsize=(10,10))
corr = df_sample_all.corr()
ax = sns.heatmap( corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
    square=True, annot=True
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,);

# plot 2 

fig, axs = plt.subplots(6,sharex=True, sharey=True)

fig.suptitle('Vertically stacked subplots')
activitycolors={0:'red',1:'orange',2:'green',3:'blue',4:'red',5:'grey'}
#Visualization
for act in activity_types: 
    axs[activity_codes[act]].set_title(activity_types[activity_codes[act]])
    dfp = df_all[(df_all['sub_num']=='15') & (df_all['activity']==activity_codes[act])]
    str1= 'tab:' 
    str2 =  activitycolors[activity_codes[act]] 
    vcolor =  str1+str2 
    axs[activity_codes[act]].plot(dfp['userAcceleration.x'][:500],vcolor)



#Value Distribution of the recorded Sensor Data
#df_all[['userAcceleration.x','userAcceleration.y','userAcceleration.z']].describe()  
#df_all.columns[2:-2]

#plot 3
#df_all[['userAcceleration.x','userAcceleration.y','userAcceleration.z']].boxplot()
#fig = plt.gcf()
#fig.set_size_inches(15, 10)




### Length of time series
#series_length = list()
#for act in activity_types:
#    print(act)
#    for sub in range(1,25):
#        sub = str(sub)
#        print(sub)
#        series_length.append(df_all[(df_all['sub_num']==sub) & (df_all['activity']==activity_codes[act])].shape[0])
#plt.figure(2)
#plt.title('Histogram of length of raw time series')
#plt.hist(series_length,rwidth=0.5,align='left')


# Plot class distribution:

#activiry_counts.plot(kind='bar', title='Activity Class Distibution')
#plt.show()



# Plot a pie chart for different activities
activiry_counts = df_all.activity.apply(lambda x: activity_types[x] ).value_counts()
activities = ['Walking Downstairs', 'Jogging', 'Sitting', 'Standing', 'Walking Upstairs', 'Walking']
plt.rcParams.update({'figure.figsize': [20, 20], 'font.size': 24})
plt.pie(activiry_counts, labels = activities, autopct = '%0.2f')


#############################################
##  TEST  TRAIN  
#############################################
dat.dtypes
df_sample_all.dtypes
df_sample_all.dtypes



 



#dfML1 = df_sample_all[["userAcceleration.x","userAcceleration.y","userAcceleration.z","rotationRate.x","rotationRate.y","rotationRate.z","activity","sub_num"]]

dfML1 = df_sample_all

dfML1.shape
dfML1.dtypes
#dfML["subnum"] =  dfML["sub_num"].astype(str).astype(int)
dfML1["subnum"] =  dfML1["sub_num"].astype(str)
dat["code"] = dat["code"].astype(str)

pd.merge(dfML1, dat, left_on='sub_num', right_on='code', how='inner').drop('code', axis=1)
dfPersons =  pd.merge(dfML1, dat, left_on='sub_num', right_on='code', how='inner').drop('code', axis=1)
dfPersons.head(2)
dfPersons.describe()
dfPersons.dtypes


dfML = dfPersons[["userAcceleration.x","userAcceleration.y","userAcceleration.z","rotationRate.x","rotationRate.y","rotationRate.z","gravity.x","gravity.y","gravity.z","height","age","gender","activity"]]
array = dfML.values

X = array[:,0:12]
y = array[:,12]
target =y
 
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

#############################################
##     CLUSTERING PCA        K-MEAN 
#############################################
## Clustering the data activities.
## Getting the set of the measurments figures
arrayclu = df_all.values
Xclu = arrayclu[:,0:12]


# Using the elbow method to find out the optimal number of #clusters. 
#KMeans class from the sklearn library.
from sklearn.cluster import KMeans
wcss=[]
#this loop will fit the k-means algorithm to our data and 
#second we will compute the within cluster sum of squares and #appended to our wcss list.
for i in range(1,11): 
     kmeans = KMeans(n_clusters=i, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )
#i above is between 1-10 numbers. init parameter is the random #initialization method  
#we select kmeans++ method. max_iter parameter the maximum number of iterations there can be to 
#find the final clusters when the K-meands algorithm is running. we #enter the default value of 300
#the next parameter is n_init which is the number of times the #K_means algorithm will be run with
#different initial centroid.
     kmeans.fit(Xclu)
#kmeans algorithm fits to the X dataset
     wcss.append(kmeans.inertia_)
#kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
#4.Plot the elbow graph
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()




 

 # k-means clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
# define dataset
labels = ['jog','dws','sit','std', 'ups', 'wlk']
# define dataset
Xclu, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = KMeans(n_clusters=6)
# fit the model
model.fit(Xclu)
# assign a cluster to each example
yhat = model.predict(Xclu)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
# get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
# create scatter of these samples
    pyplot.scatter(Xclu[row_ix, 0],Xclu[row_ix, 1] ,label = labels[cluster])
# show the plot
plt.legend()
plt.xlabel('Categorising activities')
plt.title('Segmentation K-Means')
plt.show()
pyplot.show()



#############################################
##  MODELING 
#############################################


# visualize the data
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn

 



results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    #kfold = = KFold(len(y), n_folds=10, shuffle=True, random_state=0)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
    
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


##from sklearn.ensemble import KNeighborses
# Make predictions on validation dataset using KNeighborsClassifier 
modelKNN = KNeighborsClassifier()
#model = SVC(gamma='auto')
modelKNN.fit(X_train, Y_train)
predictions = modelKNN.predict(X_validation)

#conda install -c anaconda ipython
#pip install -U scikit-learn scipy matplotlib

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
from sklearn.metrics import classification_report
target_names = ['Walking Downstairs', 'Jogging', 'Sitting', 'Standing', 'Walking Upstairs', 'Walking']
print(classification_report(Y_validation, predictions, target_names=target_names))
#print(classification_report(Y_validation, predictions))  

# Creating a Confusion Matrix
class_names = ['Walking Downstairs', 'Jogging', 'Sitting', 'Standing', 'Walking Upstairs', 'Walking']
cm = confusion_matrix(Y_validation, predictions)
#df_cm = pd.DataFrame(cm, index = (2, 2), columns = (0, 5))
df_cm = pd.DataFrame(cm  )
plt.figure(figsize = (28,20))
fig, ax = plt.subplots()
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
#class_names=[0,5]
tick_marks = np.arange(len(class_names))
plt.tight_layout()
plt.title('Confusion Matrix For KNN Algorithm\n', y=1.1)
plt.xticks(tick_marks, class_names,rotation = 45)
plt.yticks(tick_marks, class_names,rotation = 0)
ax.xaxis.set_label_position("top")
plt.ylabel('Actual label\n')
plt.xlabel('Predicted label\n')

for i in range(len(Y_validation)):
   print ("Y_validation :   ", Y_validation[i]," --> predictions  ", predictions[i])
   
# summarize the first 5 cases
for i in range(5):
    print('%s => %d (expected %d)' % (X_validation[i].tolist(), predictions[i], Y_validation[i]))




#Each row of the matrix represents the predicted class.
#Each column of the matrix represents the actual class  

#Compute precision, recall, F-measure and support for each class.

#The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

#The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

#The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.

#The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.

#The support is the number of occurrences of each class in y_true.


############### Deep Learning 
 
dfDL = dfPersons[["userAcceleration.x","userAcceleration.y","userAcceleration.z","rotationRate.x","rotationRate.y","rotationRate.z","gravity.x","gravity.y","gravity.z","height","age","gender","activity"]]
X = dfDL.iloc[:,:12].values
y = dfDL.iloc[:,12:13].values


 #Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)



# Import `StandardScaler` from `sklearn.preprocessing`
# from sklearn.preprocessing import StandardScaler
# Define the scaler 
# scaler = StandardScaler().fit(X_train)

# Scale the train set
# X_train = scaler.transform(X_train)
# Y_train = scaler.transform(Y_train)
# Scale the test set
# X_validation = scaler.transform(X_validation)
# Y_validation = scaler.transform(Y_validation)


# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model_dp1  = Sequential()
# Add an input layer 
model_dp1 .add(Dense(13, activation='relu', input_shape=(12,)))
# Add one hidden layer 
model_dp1 .add(Dense(8, activation='relu'))
# Add an output layer 
#model_dp1 .add(Dense(1, activation='sigmoid'))
model_dp1 .add(Dense(6, activation='softmax'))


# Model summary
model_dp1 .summary()
# Model config
model_dp1 .get_config()
# List all weight tensors 
model_dp1 .get_weights()

model_dp1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

# training the model
from time import time
##t0 = time()
history = model_dp1.fit(X_train, Y_train,
 validation_data = (X_validation,Y_validation),
 epochs=10,
 batch_size=64) 


 y_pred = model_dp1.predict(X_validation)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
     pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(Y_validation)):
     test.append(np.argmax(Y_validation[i]))
 
 
for i in range(5):
    print(' Prediction %d ==> (expected: %d)' % (pred[i], test[i]))
    #print('%s => %d (expected %d)' % (X_validation[i].tolist(), pred[i],
       
  