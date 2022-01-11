# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:23:11 2021

@author: Filippos
"""
######################################
# Import Libraries 
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
# Data Import  - File from 24 participants 
##################################
dat = pd.read_csv('C:/Users/Filippos/Desktop/Master/data_subjects_info.csv')
dat.head()

##################################
## Data Overview 
##################################
dat.dtypes
dat.head()
dat.info()
dat.describe()
dat.isna().sum()
#################################
## Data Analysis    
#################################
## to see the corelattion of the columns
 
dat.corr()
## corr_matrix = dat.corr()
## corr_matrix["age"].sort_values(ascending=False)
## corr_matrix["height"].sort_values(ascending=False)
## dat.groupby(["gender"]).count()
dat.groupby(["gender"]).size()

#################################
## Data Visualization
#################################

# Plot 1
## Correlation
plt.figure(figsize=(10,10))
temp = dat.copy()
temp=temp.drop(columns=['code'])
corr = temp.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0,    cmap=sns.diverging_palette(20, 220, n=200),    square=True, annot=True)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45, horizontalalignment='right')
ax.set_yticklabels(ax.get_yticklabels(),  rotation=45);
plt.title('Correllation Graph of Participants')

# Plot 2
import warnings
warnings.filterwarnings('ignore')
temp = dat.copy()
temp=temp.drop(columns=['code'])
sns.pairplot(temp, size=3)

# Plot 2.1
temp=dat
temp=temp.drop(columns=['code'])
sns.pairplot(temp,hue='age', size=3)

# Plot 3 - Relation of Wight vs Height of Participants
sns.regplot(x=dat['weight'], y=dat['height'])
plt.title('Relation of Wight vs Height of Participants')

# Plot 4 - Age of Participants
dat['age'].value_counts().plot(kind='bar', title='Age',figsize=(20,8)) 
plt.xlabel('Age of Participants')
plt.ylabel('Number of occurance')
plt.title('Graph showing the Age of Participants')

#Plot 5 - Mesurments 
dat[['weight','height','age']].boxplot()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.xlabel('Field of Participants')
plt.ylabel('Measurments')
plt.title('Box Plot showing the Age,Weight,Height of Participants')

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
    #print('j',j)
    csv = glob(j + '/*')
    for i in csv:
        #print('load file of participant',i)
        df = pd.read_csv(i)
        ## df['activity'] = activity_codes[j[68:71]]
        st= j[-8:]
        p= st.find('\\') 
        st = st[p+1:p+4]
        df['activity'] = activity_codes[st]  
        df['sub_num'] = i[len(j)+5:-4]
        ##df['activity'] = activity_codes[j[68:71]]
        ##df['sub_num'] = i[len(j)+5:-4]
        df_all_list.append(df)
        df_sample.append(df.sample(n=300))
               
print('All Files 16X24 are loadeded')

df_all = pd.concat(df_all_list,axis=0)
df_all = df_all.drop('Unnamed: 0',axis=1)
df_sample_all = pd.concat(df_sample,axis =0)
df_sample_all = df_sample_all.drop('Unnamed: 0',axis=1)

################################################
# Data Overview of the measurments of activities
################################################
df_all.head()
#Check for null values in the fields
df_all.isna().sum()
df_sample_all.head()
df_sample_all.describe()
df_sample_all.head()
df_sample_all.isna().sum()


## Corellation 
plt.figure(figsize=(10,10))
corr = df_sample_all.corr()
ax = sns.heatmap( corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200),
    square=True, annot=True
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,);

# Vertically stacked subplots
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
    axs[activity_codes[act]].plot(dfp['userAcceleration.y'][:500],vcolor)

# Plot a pie chart for different activities
import warnings
warnings.filterwarnings('ignore')
activiry_counts = df_all.activity.apply(lambda x: activity_types[x] ).value_counts()
activities = ['Walking Downstairs', 'Jogging', 'Sitting', 'Standing', 'Walking Upstairs', 'Walking']
plt.rcParams.update({'figure.figsize': [10, 10], 'font.size': 24})
plt.pie(activiry_counts, labels = activities, autopct = '%0.2f')
plt.title('Distribution of Activities Measurments')

# Merging Files 
dfML1 = df_sample_all
dfML1["subnum"] =  dfML1["sub_num"].astype(str)
dat["code"] = dat["code"].astype(str)

pd.merge(dfML1, dat, left_on='sub_num', right_on='code', how='inner').drop('code', axis=1)
dfPersons =  pd.merge(dfML1, dat, left_on='sub_num', right_on='code', how='inner').drop('code', axis=1)
print(('Merging files regarding participants and activities is completed'))

dfPersons.head(5)
dfPersons.describe()

##########################################################################
## PCA (Principle Component Analysis)
##########################################################################
df=df_all
df = df.loc[~df.index.duplicated(keep='first')]

features = ['attitude.roll', 'attitude.pitch', 'attitude.yaw', 'gravity.x','gravity.y','gravity.z','rotationRate.x','rotationRate.y','rotationRate.z','userAcceleration.x','userAcceleration.y','userAcceleration.z']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['activity']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

## principalDf.shape
finalDf=[]
target = df[['activity']]
## target.shape
principalDf.isna().sum()
## target.isna().sum()

finalDf=[]
finalDf = pd.concat([principalDf, target], axis = 1)
## finalDf.head(5)
## finalDf.dtypes
##vfinalDf.isna().sum()

fig = plt.figure(figsize = (16,16))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
 
## {'dws':0,'jog':1,'sit':2,'std':3,'ups':4,'wlk':5}
labels =['Walking Downstairs', 'Jogging', 'Sitting', 'Standing', 'Walking Upstairs', 'Walking']
targets =  [0,1,2,3, 4, 5]
colors  =  ['r','g','b','c','m','y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['activity'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
    
ax.legend(labels)
ax.grid()

###################################
## Clustering the data activities
###################################

##  Getting the set of the measurments figures
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
#kmeans algorithm fits to the X dataset
     wcss.append(kmeans.inertia_)
#kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
#4.Plot the elbow graph
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


##############################################
###  CLUSTERING 
##############################################

# k-means clustering
from matplotlib import pyplot
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
##lgd = ax.legend(loc='lower right')
labels = ['dws','jog','sit','std', 'ups', 'wlk']
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

#########################################################3
###  TRAIN / TEST SET
#########################################################
dfML = dfPersons[["userAcceleration.x","userAcceleration.y","userAcceleration.z","rotationRate.x","rotationRate.y","rotationRate.z","gravity.x","gravity.y","gravity.z","height","age","gender","activity"]]
array = dfML.values

X = array[:,0:12]
y = array[:,12]
target =y

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
print('train set and validation set is completed')

######################################################
##   MODEL
###################################################

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from time import time

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LR2',LogisticRegression(penalty='l2',solver = 'lbfgs', C = 1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier(criterion = 'entropy', random_state = 42)))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
import warnings
warnings.filterwarnings('ignore')
print('Processing models...')
results = []
names = []
for name, model in models:
     
    t0 = time()
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    f = '{0:.2f}'.format(time()-t0) 
    print('Training: %s: %f (%f) takes %s seconds' % (name, cv_results.mean(), cv_results.std(),  f))
    

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Model algorithm Comparison')
pyplot.ylabel('Accuracy')
pyplot.show()

#########################################
###   PREDICTION & EVALUATION 
#########################################

## Building a Random Forest Model 
rfc = RandomForestClassifier(criterion = 'entropy', random_state = 42)
rfc.fit(X_train, Y_train)
print('Random Forest Model fit completed')

# Prediction of test set using the RFC model
predictionsrfc =rfc.predict(X_validation)
print('  prediction completed')

# Evaluation of the Random Forest Model
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Evaluate predictions
print('accuracy_score \n',accuracy_score(Y_validation, predictionsrfc))
print('\n confusion_matrix \n',confusion_matrix(Y_validation, predictionsrfc))
print('\n classification_report \n',classification_report(Y_validation, predictionsrfc))

# Creating a Confusion Matrix

target_names = ['Walking Downstairs', 'Jogging', 'Sitting', 'Standing', 'Walking Upstairs', 'Walking']
class_names = ['Walking Downstairs', 'Jogging', 'Sitting', 'Standing', 'Walking Upstairs', 'Walking']
cm = confusion_matrix(Y_validation, predictionsrfc)
#df_cm = pd.DataFrame(cm, index = (2, 2), columns = (0, 5))
df_cm = pd.DataFrame(cm  )
plt.figure(figsize = (28,20))
fig, ax = plt.subplots()
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
#class_names=[0,5]
tick_marks = np.arange(len(class_names))
plt.tight_layout()
plt.title('Confusion Matrix For Random Forest Model  Algorithm\n', y=1.1)
plt.xticks(tick_marks, class_names,rotation = 45)
plt.yticks(tick_marks, class_names,rotation = 0)
ax.xaxis.set_label_position("top")
plt.xlabel('Actual label\n')
plt.ylabel('Predicted label\n')

from sklearn.metrics import classification_report
print('\n classification_report \n',classification_report(Y_validation, predictionsrfc))

# summarize the first 5 cases
for i in range(5):
    print('%s => Predicted: %d (Expected: %d)' % (X_validation[i].tolist(), predictionsrfc[i], Y_validation[i]))


## Building also DecisionTreeClassifier to compare with the  Random Forest Model 
dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dt.fit(X_train, Y_train)
dt_pred_train = dt.predict(X_train)

# Comparison between Decision Tree and Random Forest
feature_importance=pd.DataFrame({
    'dt':dt.feature_importances_,
     'rfc':rfc.feature_importances_
},index=dfML.drop(columns=['activity']).columns)
feature_importance.sort_values(by='dt',ascending=True,inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18,8))
dt_feature=ax.barh(index+0.4,feature_importance['dt'],0.4,color='lightgreen',label='Decision Tree')
rfc_feature=ax.barh(index,feature_importance['rfc'],0.4,color='purple',label='Random Forest')
ax.set(yticks=index+0.4,yticklabels=feature_importance.index)

ax.legend()
plt.show()

############################################################
##   Deep Learning Analysis
#############################################################
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

model_dp1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['categorical_accuracy'])

 
history = model_dp1.fit(X_train, Y_train,  
                        validation_data = (X_validation,Y_validation),
                        epochs=10, 
                        batch_size=64)    
#### Prediction
y_pred = model_dp1.predict(X_validation)
#Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in  range(len(Y_validation)):
    test.append(np.argmax(Y_validation[i]))

for i in range(5):
    print(' Prediction  %d   ==> (expected:  %d)' % (pred[i], test[i]))


# Model Accuracy between Train and Test
import matplotlib.pyplot as plt
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
