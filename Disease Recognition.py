#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Audio data analysis has facilitated the route of identifying voice disorders in a non-invasive manner. As vocal cord malady 
#can immerse anytime from various bad habits (loud sound, smoking, extra force on vocal cords, etc.) and neurological 
#imbalance, early recognition of voice disorders can save people from causing long-term damage. In this research, the three most
#injurious voice diseases have been recognized exploiting the LDA based MFCC feature matrix and KNN algorithm. 


# In[2]:


# https://github.com/Shovan5795/Linear-Discriminant-Analysis-based-Voice-Disease-Recognition


# In[68]:


import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt 
import glob
import os
import numpy as np
import pandas as pd


# In[124]:


final_voice_dys =[]
col=0
no_of_mfcc = 13


# In[125]:


pathd = r"E:\\Education\\sem 10\\Data\\Dysophia"


# In[126]:


for filename in glob.glob(os.path.join(pathd, '*.wav')):
    datad, sample_rated = librosa.load(filename)
    mfccsd = librosa.feature.mfcc(y=datad, sr=sample_rated, n_mfcc=no_of_mfcc)
    mfccs1d = np.array(mfccsd)
    voice_dys = mfccs1d.flatten()
    final_voice_dys.append(voice_dys)

dataset1 = pd.DataFrame(final_voice_dys)
dysphon = dataset1.to_csv(r'E:\\Education\\sem 10\\Data\\dysphon.csv', index = False)


# In[127]:


path2 = r"E:\\Education\\sem 10\\Data\\Healthy"


# In[128]:


final_healthy =[]

for filename in glob.glob(os.path.join(path2, '*.wav')):
    datah, sample_rateh = librosa.load(filename)
    mfccsh = librosa.feature.mfcc(y=datah, sr=sample_rateh, n_mfcc=no_of_mfcc)
    mfccs1h = np.array(mfccsh)
    voice_healthy = mfccs1h.flatten()
    final_healthy.append(voice_healthy)

dataset2 = pd.DataFrame(final_healthy)

healthy = dataset2.to_csv(r'E:\\Education\\sem 10\\Data\\healthy.csv', index = False)


# In[129]:


path3="E:\\Education\\sem 10\\Data\\Reinkes"
final_rank =[]

for filename in glob.glob(os.path.join(path3, '*.wav')):
    datar, sample_rater = librosa.load(filename)
    mfccsr = librosa.feature.mfcc(y=datar, sr=sample_rater, n_mfcc=no_of_mfcc)
    mfccs1r = np.array(mfccsr)
    voice_rank = mfccs1r.flatten()
    final_rank.append(voice_rank)

dataset4 = pd.DataFrame(final_rank)

rank = dataset4.to_csv(r'E:\\Education\\sem 10\\Data\\ranke.csv', index = False)


# In[130]:


path4="E:\\Education\\sem 10\\Data\\Laryngitis"
final_lar =[]

for filename in glob.glob(os.path.join(path4, '*.wav')):
    datal, sample_ratel = librosa.load(filename)
    mfccsl = librosa.feature.mfcc(y=datal, sr=sample_ratel, n_mfcc=no_of_mfcc)
    mfccs1l = np.array(mfccsl)
    voice_lar = mfccs1l.flatten()
    final_lar.append(voice_lar)

dataset3 = pd.DataFrame(final_lar)

lar = dataset3.to_csv(r'E:\\Education\\sem 10\\Data\\laringitis.csv', index = False)


# In[131]:


final_dataset = pd.concat([dataset1, dataset2,dataset3, dataset4], axis = 0).to_csv(r'E:\\Education\\sem 10\\Data\\final.csv', index = False)


# In[132]:


df = pd.read_csv(r"E:\\Education\\sem 10\\Data\\final.csv")


# In[133]:


df.shape


# In[134]:


df.size


# In[135]:


final_preprocessed_data = df.replace(np.nan, 0)
#final preprocessed_data.hist()
#plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_final_preprocessed_data = scaler.fit_transform(final_preprocessed_data) 
scaled_final_preprocessed_data = np.array(scaled_final_preprocessed_data)


final_preprocessed_data["Label"] = 0
final_preprocessed_data.iloc[0:52, 2769:2770] = 1  #Dysphonia
final_preprocessed_data.iloc[52:192, 2769:2770] = 0 #Healthy
final_preprocessed_data.iloc[192:315, 2769:2770] = 2 #Laryngitis
final_preprocessed_data.iloc[315:368, 2769:2770] = 3 #Rankei Odema

aggregated_dataset = final_preprocessed_data.to_csv(r"E:\\Education\\sem 10\\Data\\final_version.csv")


# In[136]:


import pandas as pd
df = pd.read_csv(r"E:\\Education\\sem 10\\Data\\final_version.csv")

X = df.iloc[0:368,0:2770]
y = df.iloc[:,-1]


# In[137]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_new = scaler.fit_transform(X)


# In[138]:


#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_PCA2= pca.fit_transform(X_new)


# In[139]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', n_init = 10, max_iter=300, random_state=0)
ykmeans = kmeans.fit(X_new)

#K-NN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=4)
ymr = neigh.fit(X_new, y)


# In[140]:


#SVM
from sklearn import svm
#ymr = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_PCA, y)
ymr = svm.SVC(kernel='poly', degree=3, C=1).fit(X_PCA2, y)
#DT
from sklearn.tree import DecisionTreeClassifier
ymr = DecisionTreeClassifier().fit(X_PCA2, y)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
ymr = LogisticRegression(solver = 'liblinear', multi_class = 'ovr').fit(X_PCA2, y)


# In[141]:


from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, cross_val_score
kf = KFold(n_splits = 10)
skf = StratifiedKFold(n_splits = 10)
shs = ShuffleSplit(n_splits = 10)
sshs = StratifiedShuffleSplit(n_splits = 10)

import datetime
start_time = datetime.datetime.now()
acc1 = cross_val_score(ymr, X_new, y, scoring = 'accuracy', cv=kf, n_jobs=1)
acc2 = cross_val_score(ymr, X_new, y, scoring = 'accuracy', cv=skf, n_jobs=1)
acc3 = cross_val_score(ymr, X_new, y, scoring = 'accuracy', cv=shs, n_jobs=1)
acc4 = cross_val_score(ymr, X_new, y, scoring = 'accuracy', cv=sshs, n_jobs=1)
a1 = acc1.mean()
a2 = acc2.mean()
a3 = acc3.mean()
a4 = acc4.mean()
end_time = datetime.datetime.now()
time_diff = (end_time - start_time)
execution_time = (time_diff.total_seconds() * 1000)/4


# In[142]:


print(acc1,acc2)


# In[143]:


import matplotlib.pyplot as plt
plt.scatter(X_PCA2[y == 0, 0], X_PCA2[y == 0, 1], s=100, c='blue', label = 'Healthy')
plt.scatter(X_PCA2[y == 1, 0], X_PCA2[y == 1, 1], s=100, c='green', label = 'Dysphonia')
plt.scatter(X_PCA2[y == 2, 0], X_PCA2[y == 2, 1], s=100, c='red', label = 'Laryngitis')
plt.scatter(X_PCA2[y == 3, 0], X_PCA2[y == 3, 1], s=100, c='cyan', label = 'Reinkes Edema')
plt.title("Scatter Plot Diagram for Different Voice Classes")
plt.xlabel("1st PCA of MFCC Feature")
plt.ylabel("2nd PCA of MFCC Feature")
plt.legend()
plt.show()


# In[ ]:


X = df.iloc[0:368,0:2770]
y = df.iloc[:,-1]


# In[144]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_new = scaler.fit_transform(X)


# In[145]:


from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_PCA2= pca.fit_transform(X_new)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import MaxPool1D


# In[146]:


def create_model():
    model = Sequential()
    model.add(Dense(256, activation = 'relu', input_shape = (367,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss='sparse_categorical_crossentropy',optimizer= Adam(lr=0.0001),metrics=['accuracy'])
    print(model.summary())
    return model

history = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 10)


# In[147]:


from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit, cross_val_score

kf = KFold(n_splits = 10)
skf = StratifiedKFold(n_splits = 10)
shs = ShuffleSplit(n_splits = 10)
sshs = StratifiedShuffleSplit(n_splits = 10)

import datetime

start_time = datetime.datetime.now()
acc1 = cross_val_score(history, X_PCA2, y, scoring = 'accuracy', cv=kf, n_jobs=1)
acc2 = cross_val_score(history, X_PCA2, y, scoring = 'accuracy', cv=skf, n_jobs=1)
acc3 = cross_val_score(history, X_PCA2, y, scoring = 'accuracy', cv=shs, n_jobs=1)
acc4 = cross_val_score(history, X_PCA2, y, scoring = 'accuracy', cv=sshs, n_jobs=1)
print(acc1.mean())
print(acc2.mean())
print(acc3.mean())
print(acc4.mean())
end_time = datetime.datetime.now()
time_diff = (end_time - start_time)
execution_time = (time_diff.total_seconds() * 1000)/4


# In[ ]:




