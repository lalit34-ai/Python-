#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import nltk
from nltk.corpus import cmudict
import re
import string
import librosa.display
from nltk.corpus import stopwords
from collections import Counter


# In[2]:


#import Ipython
#from glob import tqdm
#import seaborn as sns


# In[3]:


from wordcloud import WordCloud,STOPWORDS


# In[4]:


import gc
import librosa 
import IPython.display  as ipd
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# In[5]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, Input
from keras.optimizers import Adam
from keras import backend as K


# In[6]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# In[56]:


#Paper: Dysarthic Speech Recognition Using a Deep-BiDirectional LSTM(and CTC decoder)

# Here we train a SRS on dysarthic using DBLSTM and CTC decoder,we have selected DBLSTM coz of the inconsistency of acoustic cues in speech patterns
# so that its memory cell can handle inconsistent temporal behaviour.

path="C:\\Users\\lalit\\Downloads\\UASpeech_original_FM-006\\Speech\\UASpeech\\audio\\original"
def load_audio_files(path):
    audio_files=[]
    labels=[]
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                file_path=os.path.join(root,file)
                label=os.path.basename(file)
                audio_files.append(file_path)
                labels.append(label)
    return audio_files,labels


# In[57]:


audio_files,labels=load_audio_files(path)


# In[77]:


def extract_features(audio_files):  # extracting the features
    features=[]
    for file in audio_files:
        y,sr=librosa.load(file,mono=True,sr=None)
        #y=tf.squeeze(y,axis=1)
        sr=tf.cast(sr,dtype=tf.int64)
        if(len(y)<1024):
            continue
        mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mels=128,n_fft=512,hop_length=256)
        features.append(np.mean(mfcc.T, axis=0))
    return features


# In[80]:


y,sr=librosa.load(audio_file,mono=True,sr=None)


# In[78]:


print(f"Total audio files loaded: {len(audio_files)}")


# In[2]:


# features = extract_features(audio_files)


# In[ ]:


data=pd.DataFrame(features)
data['labels']=labels
data.to_csv('mfcc_features.csv', index=False)


# In[ ]:


y,sr=librosa.load(audio_files[0],mono=True,sr=None)
mfcc=librosa.feature.mfcc(y=y,sr=sr)
plt.figure(figsize=(10,4))
librosa.display.specshow(mfcc,x_axis='time',sr=sr)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()


# In[ ]:


mel = librosa.feature.melspectrogram(y=y, sr=sr)  # MelSpectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-spectrogram')
plt.tight_layout()
plt.show()


# In[ ]:


chroma = librosa.feature.chroma_stft(y=y, sr=sr)  #chroma Feature
plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chroma Feature')
plt.tight_layout()
plt.show()


# In[ ]:


# Spectral contrast
contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(contrast, x_axis='time')
plt.colorbar()
plt.title('Spectral Contrast')
plt.tight_layout()
plt.show()


# In[ ]:


# Tonnetz
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
plt.figure(figsize=(10, 4))
librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time')
plt.colorbar()
plt.title('Tonnetz')
plt.tight_layout()
plt.show()


# In[ ]:


data = pd.DataFrame(features)
data['label'] = labels
data.to_csv('extended_features.csv', index=False)

X = data.drop('label', axis=1)
y = data['label']


# In[19]:


#import extended_features


# In[5]:


data=pd.read_csv("C:\\Users\\lalit\\Downloads\\extended_features.csv",nrows=50000)
data.head()


# In[6]:


data.dtypes


# In[7]:


# X = data.drop('label', axis=1)
X = data.iloc[: ,:-1].values
y = data['label'].values


# In[10]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(y).reshape(-1,1)).toarray()


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[8]:


#LabelEncoder
from sklearn.preprocessing import LabelEncoder
convertor = LabelEncoder()


# In[9]:


y=convertor.fit_transform(y)


# In[3]:


import vaex


# In[10]:


# X = np.array((X-np.min(X))/(np.max(X)-np.min(X)))
# X = X/np.std(X)
# y = np.array(y)

from sklearn.preprocessing import StandardScaler
fit=StandardScaler()
X=fit.fit_transform(np.array(X,dtype=float))


# In[8]:


X_train ,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)


# In[29]:


# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)


# In[ ]:


# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
# #Print the shapes
# X_train.shape, X_test.shape, X_val.shape, len(y_train), len(y_test), len(y_val) 
#Extract training, test and validation datasets


# In[49]:


X_train[1]


# In[50]:


X_train.shape[1]


# In[31]:


X_test.shape


# In[32]:


y_train.shape


# In[33]:


y_test.shape


# In[35]:


len(y_test),len(y_train)


# In[44]:


# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train=scaler.fit(X_train).transform(X_train)
# X_test=scaler.fit(X_test).transform(X_test)

def trainModel(model,epochs,optimizer):
    batch_size =128
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics='accuracy'
    )
    return model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epochs,batch_size=batch_size)


# In[45]:


def plotValidate(history):
    print("Validation Accuracy",max(history.history["val_accuracy"]))
    pd.Dataframe(history.history).plot(figsize=(12,6))
    plt.show()
# I apply Principal Component Analysis on the dataset


# In[54]:


import keras as k


# In[55]:


model=k.models.Sequential([
    k.layers.Dense(512,activation='relu',input_shape=(X_train.shape[1],)),
    k.layers.Dropout(0.2),
    k.layers.Dense(256,activation='relu'),
    k.layers.Dropout(0.2),
    k.layers.Dense(128,activation='relu'),
    k.layers.Dropout(0.2),
    k.layers.Dense(64,activation='relu'),
    k.layers.Dropout(0.2),
    k.layers.Dense(50000,activation='softmax'),   
])
print(model.summary())
model_history=trainModel(model=model,epochs=600,optimizer='adam')


# In[53]:


test_loss, test_acc = model.evaluate(X_test,y_test,batch_size=128)
print("The test Loss is:",test_loss)
print("\nThe Best test Accuracy is:",test_acc*100)


# In[16]:


X_train.shape # for each 51765 video we have 20 dimensions


# In[12]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[13]:


data.info()


# In[14]:


X_train.shape 


# In[ ]:


rfc = RandomForestClassifier(n_estimators=2)  # RandomForest Classifier
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)


# In[ ]:


print("Random Forest Classifier accuracy:", accuracy_score(y_test, y_pred_rfc))


# In[ ]:


ada = AdaBoostClassifier(n_estimators=100, random_state=42)     # Ada Boost Classifier
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)

gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)  # Gradient Boost Classifier
gbc.fit(X_train, y_train)
y_pred_gbc = gbc.predict(X_test)


# In[ ]:


print("AdaBoost Classifier accuracy:", accuracy_score(y_test, y_pred_ada))
print("Gradient Boosting Classifier accuracy:", accuracy_score(y_test, y_pred_gbc))

accuracies = [accuracy_score(y_test, y_pred_rfc), accuracy_score(y_test, y_pred_ada), accuracy_score(y_test, y_pred_gbc)]
labels = ['Random Forest', 'AdaBoost', 'Gradient Boosting']
plt.bar(labels, accuracies)
plt.xlabel('Ensemble Method')
plt.ylabel('Accuracy')
plt.title('Ensemble Methods Accuracy Comparison')
plt.show()


# In[74]:


from tensorflow.keras.layers import Embedding
# embedding_vecor_length = 32
# model = Sequential()
# model.add(Embedding(X_train, embedding_vecor_length, input_length=y_train))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)


# In[7]:


#print(dir(X_train)) # to check if the data or object is iterable or not


# In[10]:


input_shape=(26250,20)
model = keras.Sequential()    # try range
model.add(LSTM(256,input_shape=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(48, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(50000, activation='softmax'))
model.summary()


# In[30]:


# The algorithm takes input 1 audio file of 1 word spoken. Its passed through DBLSTM, the output is a phonetic transription of word.
# each subject was asked to recite a total of 756 words.
#The words in dataset consiste of digits(0,1 etc),the International radio alphabet(the,of,) and uncommon words(naturalization,frugality etc)
# the data is split randomly in a ratio of 84-8-8.
# before giving to BDLSTm preprocessing is done on it(normalized filter banks are foud by appliying pre-emphasis filter,fourier transfrm,triangle filter,normalization for time windows of 10ms.)

#Eg fig1 audio signal and its normalized filter bank for word paste


# In[31]:


filename="E:\\Education\\sem 10\\recordings\\test"
plt.figure(figsize=(12,4)) # only for a selected folder
for file in os.listdir(filename):
    print(file)


# In[32]:


#Extracting Features from Data
ipd.Audio("C:\\Users\\lalit\\Downloads\\UASpeech_original_FM-006\\Speech\\UASpeech\\audio\\original\\M01\\M01_B2_CW91_M6.wav")


# In[33]:


def feature_extraction(file):
    x,sample_rate=librosa.load(file,res_type="kaiser_fast")
    mfcc=np.mean(librosa.feature.mfcc(y=x,sr=sample_rate,n_mfcc=100).T,axis=0)
    return mfcc


# In[34]:


df=pd.read_csv("E:\\Education\\sem 10\\overview-of-recordings.csv")


# In[35]:


df.head()


# In[36]:


df.isnull().sum()


# In[37]:


from nltk.stem import PorterStemmer
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# In[38]:


cv=CountVectorizer()
cv.fit(df['phrase'])
result=cv.transform(df['phrase'])
print(result.shape)


# In[39]:


# prompt : it tells us the medical problem category
# phrase : it tells us the text speaker spoke
# filename : it tells us the name of the file    # we will be taking these 3 things into account while training the data


# In[40]:


cv1=CountVectorizer()
cv1.fit(df['prompt'])
result1=cv1.transform(df['prompt'])
print(result.shape)


# In[41]:


df['prompt'].value_counts()


# In[42]:


import seaborn as sns
plt.figure(figsize=(10, 6))
sns.countplot(df['prompt'])
plt.title("Count of records in each class")
plt.xticks(rotation="vertical")
plt.show()


# In[43]:


def EDA1(df):
    print(f"The number of rows are {df.shape[0]} and {df.shape[1]} columns." )
    print(df.info())
EDA1(df)


# In[44]:


# We check for duplicates:
def remove_duplicates(df):
    dup=df.duplicated().sum()
    print("The number of duplicates is",dup)
    if(dup>=1):
        df.drop_duplicates(inplace=True)
    else:
        print("No Duplicates found in data")
remove_duplicates(df)


# In[45]:


df1=df[['phrase','prompt']]
df1.head()


# In[46]:


# !pip install spacy
#import spacy
#from spacy.lang.en.stop_words import STOP_WORDS


# In[47]:


def clean_txt(docs):
    lemmatizer = WordNetLemmatizer() 
    # split into words
    speech_words = nltk.word_tokenize(docs)
    # convert to lower case
    lower_text = [w.lower() for w in speech_words]
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    stripped = [re_punc.sub('', w) for w in lower_text]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if not w in  list(STOPWORDS)]
    # filter out short tokens
    words = [word for word in words if len(word) > 2]
    #Stemm all the words in the sentence
    lem_words = [lemmatizer.lemmatize(word) for word in words]
    combined_text = ' '.join(lem_words)
    return combined_text

# Cleaning the text data
df1['cleaned_phrase'] = df1['phrase'].apply(clean_txt)
df1


# In[48]:


def feature_extraction(file):
    x,sample_rate=librosa.load(file,res_type="kaiser_fast")
    mfcc=np.mean(librosa.feature.mfcc(y=x,sr=sample_rate,n_mfcc=100).T,axis=0)
    return mfcc


# In[49]:


import numpy as np
from tqdm import tqdm

features=[]
for index_num,row in tqdm(df.iterrows()):
    file_name=os.path.join(os.path.abspath("E:\\Education\\sem 10\\recordings\\test")+"\\"+row["file_name"])
    final_class_label=row["prompt"]
    data=feature_extraction(file_name)
    features.append([data,final_class_label])


# In[54]:


features_df=pd.DataFrame(features,columns=['feature','prompt'])


# In[55]:


features_df.head()


# In[56]:


X=np.array(features_df['feature'].tolist())
y=np.array(features_df['prompt'].tolist())
### Label Encoding -> Label Encoder

labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
### Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[58]:


X.shape


# In[65]:


#implement an ANN model using Keras sequential API. 

### No of classes
num_labels=y.shape[1]
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(100,)))  #first layer with 100 neuron
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
#early_stopper = EarlyStopping( monitor = 'val_acc' , min_delta = 0.0005, patience = 3 )
 
# reduce_lr = ReduceLROnPlateau( monitor = 'val_loss' , patience = 2 , cooldown = 0)
 
# callbacks = [ reduce_lr , early_stopper]

# validation_split = 0.1 , verbose = 1 , callbacks = callbacks
 
train_history = model.fit( X_train , y_train, batch_size = 64, epochs = 50,validation_data=(X_test,y_test))
 
score = model.evaluate( X_test , y_test , batch_size = 64)
 
print( "Accuracy: {:0.4}".format( score[1] ))
 
print( "Loss:", score[0] )


# In[74]:


#Bi-LSTM Implementation
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
maxlen=50
time_steps=200
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(y.shape[1],100)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with your preferred optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on your voice data
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)


# In[70]:


# def build_model():
#     model = tf.keras.Sequential()

#     model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
#     model.add(LSTM(64))
    
#     model.add(Dense(64, activation='relu'))
#     model.add(Dropout(0.3))

#     model.add(Dense(6, activation='softmax'))

#     return model


# In[75]:


# input_shape = (13)
# model = build_model(input_shape)

# # compile model
# optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=optimiser,
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])

# model.summary()


# In[ ]:


# I got an accuracy of  50%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def classifyK_NN(features_train, labels_train):
    clf = KNeighborsClassifier()
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracyk-NN_CV",max(scores), sep=" : ")
    return clf.fit(features_train, labels_train)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def classifyLREG(features_train, labels_train):
    clf = LogisticRegression(solver='lbfgs', random_state=1)
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracyLREG_CV",max(scores), sep=" : ")
    return clf.fit(features_train, labels_train)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

def classifyNB(features_train, labels_train):
    clf = GaussianNB()
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracyNB_CV",max(scores), sep=" : ")
    return  clf.fit(features_train, labels_train)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def classifyRFC(features_train, labels_train):
    clf = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracyRFC_CV",max(scores), sep=" : ")
    return clf.fit(features_train, labels_train)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def classifySVM(features_train, labels_train):
    clf = SVC(kernel="linear")
    scores = cross_val_score(clf, features_train, labels_train, cv=10)
    print("accuracySVM_CV",max(scores), sep=" : ")
    return clf.fit(features_train, labels_train)


# Naive Bayse
clfNB = classifyNB(trainData_features, trainData_label)
predNB = clfNB.predict(testData_features)
accuracyNB = accuracy_score(predNB, testData_label)
print("accuracyNB",accuracyNB, sep=" : ")

#SVM
clfSVM = classifySVM(trainData_features, trainData_label)
predSVM = clfSVM.predict(testData_features)
accuracySVM = accuracy_score(predSVM, testData_label)
print("accuracySVM",accuracySVM, sep=" : ")

#Random Forest Classifier
clfRFC = classifyRFC(trainData_features, trainData_label)
predRFC = clfRFC.predict(testData_features)
accuracyRFC = accuracy_score(predRFC, testData_label)
print("accuracyRFC",accuracyRFC, sep=" : ")

# Linear Regression
clfLREG = classifyLREG(trainData_features, trainData_label)
predLREG = clfLREG.predict(testData_features)
accuracyLREG = accuracy_score(predLREG, testData_label)
print("accuracyLREG",accuracyLREG, sep=" : ")

# K-NN
clfK_NN = classifyK_NN(trainData_features, trainData_label)
predK_NN = clfK_NN.predict(testData_features)
accuracyK_NN = accuracy_score(predK_NN, testData_label)
print("accuracyK_NN",accuracyK_NN, sep=" : ")


# In[76]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
num_words=500
sequence_length=300
batch_size=128
model=Sequential()
e = Embedding( num_words , 10 , input_length = sequence_length )
model.add(e)
model.add(LSTM( 128 , dropout = 0.25, recurrent_dropout = 0.25))
 
model.add(Dense(1, activation = 'sigmoid' ))
 
model.summary()
 
model.compile( optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'] )
 
# early_stopper = EarlyStopping( monitor = 'val_acc' , min_delta = 0.0005, patience = 3 )
 
# reduce_lr = ReduceLROnPlateau( monitor = 'val_loss' , patience = 2 , cooldown = 0)
 
# callbacks = [ reduce_lr , early_stopper]
 
train_history = model.fit( X_train , y_train, batch_size = batch_size, epochs = 5,validation_split = 0.1 , verbose = 1 )
 
score = model.evaluate( X_test , y_test, batch_size = batch_size)
 
print( "Accuracy: {:0.4}".format( score[1] ))
 
print( "Loss:", score[0] )


# In[63]:


# from tensorflow.keras.callbacks import ModelCheckpoint
# from datetime import datetime 
# num_epochs = 40
# num_batch_size = 32
# checkpointer = ModelCheckpoint(filepath='./audio_classification.hdf5', 
#                                verbose=1, save_best_only=True)
# start = datetime.now()
# model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
# duration = datetime.now() - start
# print("Training completed in time: ", duration)


# In[ ]:


# implementing a LSTM model

model = Sequential()
model.add(LSTM(units=64, input_shape=(timesteps, features)))
model.add(Dense(units=output_dim, activation='softmax'))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=50,batch_size=32,validation_data=(X_test,y_test))


# In[ ]:


# # # ******************************************************************************************************## Other Method


# In[77]:


feature=cv.get_feature_names()
df_res=pd.DataFrame(result.toarray(),columns=feature)


# In[78]:


df_res


# In[10]:


features={}
for audio in os.listdir(filename):
    audio_path=filename+"\\"+audio
    features[audio_path]=feature_extraction(audio_path)


# In[ ]:


features[audio_path] # here I have extracted features using MFCC


# In[ ]:


extracted_featire=pd.DataFrame(features[audio_path])


# In[ ]:


X=np.array(ex)


# In[ ]:


# X=np.array(extracted_feature["feature"])
y=


# In[ ]:


# from sklearn.preprocessing import LabelEncoder
# label_encoder=LabelEncoder()
y=np.array(pd.get_dummies(y)) 


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[26]:


from itertools import cycle
from glob import glob
#sns.set_theme(style="white",palette=None)
#color_pal=plt.rcParams["axes.prop_cycle"].by_key()["color"]
#color_cycle=cycle(plt.Params["axes.prop_cycle"].by_key()["color"])


# In[ ]:


#other approach for all files


# In[28]:


audio_files= glob("C:\\Users\\lalit\\Downloads\\UASpeech_original_FM-006\\Speech\\UASpeech\\audio\\original\\*\\*.wav") # this is getting lit of all audio files


# In[29]:


y,sr=librosa.load(audio_files[0])  # raw data and sample rate


# In[30]:


y ["audio data is long numpy arrat"]


# In[31]:


sr


# In[35]:


#plot the data to get idea of what it looks like
# we will make it to pandas series
pd.Series(y).plot(figsize=(15,4),lw=1,title="Raw Audio")
plt.show()  #we see lot of disturnace, we will remove blank spaces


# In[39]:


# we trim the audio
y_trimmed,_=librosa.effects.trim(y,top_db=20)
#plot the data to get idea of what it looks like
# we will make it to pandas series #                           After Trimming
pd.Series(y_trimmed).plot(figsize=(15,4),lw=1,title="Raw Audio Trimmed",color=color_pal[1])
plt.show()  #we see lot of disturnace, we will remove blank spaces  # thisis the area with sound only


# In[40]:


# Spectrogram; to look at frequency and check how powerful they are
D=librosa.stft(y)


# In[43]:


S_db=librosa.amplitude_to_db(np.abs(D),ref=np.max)  # converting it to decibel


# In[44]:


S_db.shape


# In[50]:


# #plot the transformed data
fig,ax=plt.subplots(figsize=(15,4))
img=librosa.display.specshow(S_db,x_axis="time",y_axis="time",ax=ax)
ax.set_title("Spectrogram",fontsize=20)
fig.colorbar(img,ax=ax,format=f"%0.2f")
plt.show()


# In[52]:


# now we create Mel Spectrogram
S=librosa.feature.melspectrogram(y,sr=sr,n_mels=128,)
S.shape
S_db_mel=librosa.amplitude_to_db(S,ref=np.max)


# In[53]:


# #plot the transformed data
fig,ax=plt.subplots(figsize=(15,5))
img=librosa.display.specshow(S_db_mel,x_axis="time",y_axis="time",ax=ax)
ax.set_title("Mel Spectrogram",fontsize=20)
fig.colorbar(img,ax=ax,format=f"%0.2f")  # the audio is little accunated
plt.show()


# In[54]:


S_db_mel


# In[23]:


metadata=pd.read_csv("C:\\Users\\lalit\\Downloads\\UASpeech_original_FM-006\\Speech\\UASpeech\\doc\\Speaker.xls",encoding="cp437")


# In[21]:


with open('C:\\Users\\lalit\\Downloads\\UASpeech_original_FM-006\\Speech\\UASpeech\\doc\\Speaker.xls') as f:
    print(f)


# In[24]:


metadata.head()


# In[25]:


metadata.info()


# In[ ]:


# Loading the dysarthric speech dataset
data = pd.read_csv('dysarthric_speech.csv')

# Splitting the dataset into train and test sets
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Preprocessing the data
train_labels = train_data.pop('label')
test_labels = test_data.pop('label')
train_features = np.array(train_data)
test_features = np.array(test_data)

# Creating the LSTM model
model = keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(train_features.shape[1], 1)))
model.add(layers.Bidirectional(layers.LSTM(64)))
model.add(layers.Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fitting the model to the training data
history = model.fit(train_features.reshape(train_features.shape[0], train_features.shape[1], 1), train_labels, epochs=50, validation_split=0.2)

# Evaluating the model on the test data
test_loss, test_acc = model.evaluate(test_features.reshape(test_features.shape[0], test_features.shape[1], 1), test_labels)
print('Test accuracy:', test_acc)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatte
from tensorflow.keras.optimizers import Adam


# In[5]:


# #Original Paper
# !pip install scispacy
# !pip install pysoundfile
# !apt-get install libav-tools -y
# !apt-get install zip


# In[12]:


from fastai.text import *
from fastai.vision import *
import spacy
from spacy import displacy
import scispacy
import librosa
import librosa.display
import soundfile as sf
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import IPython
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pylab
import gc
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# !pip install fastai


# In[13]:


df=pd.read_csv("E:\Education\sem 10\overview-of-recordings.csv")
df.head()


# In[14]:


def get_wav_info(wav_file):
    data, rate = sf.read(wav_file)
    return data, rate

def create_spectrogram(wav_file):
    # adapted from Andrew Ng Deep Learning Specialization Course 5
    data, rate = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

def create_melspectrogram(filename,name):
    # adapted from https://www.kaggle.com/devilsknight/sound-classification-using-spectrogram-images
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = Path('/kaggle/working/spectrograms/' + name + '.jpg')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def wordBarGraphFunction(df,column,title):
    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])
    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
    plt.title(title)
    plt.show()

def wordCloudFunction(df,column,numWords):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    word_string=str(popular_words_nonstop)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          max_words=numWords,
                          width=1000,height=1000,
                         ).generate(word_string)
    plt.clf()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# In[15]:


overview = pd.read_csv('E:\Education\sem 10\overview-of-recordings.csv')
overview = overview[['file_name','phrase','prompt','overall_quality_of_the_audio','speaker_id']]
overview=overview.dropna()
overviewAudio = overview[['file_name','prompt']]
overviewAudio['spec_name'] = overviewAudio['file_name'].str.rstrip('.wav')
overviewAudio = overviewAudio[['spec_name','prompt']]
overviewText = overview[['phrase','prompt']]
noNaNcsv = 'E:\Education\sem 10\overview-of-recordings.csv'
noNaNcsv = pd.read_csv(noNaNcsv)
noNaNcsv = noNaNcsv.dropna()
noNaNcsv = noNaNcsv.to_csv('overview-of-recordings.csv',index=False)
noNaNcsv


# In[16]:


sns.set_style("whitegrid")
promptsPlot = sns.countplot(y='prompt',data=overview)
promptsPlot

qualityPlot = sns.FacetGrid(overview,aspect=2.5)
qualityPlot.map(sns.kdeplot,'overall_quality_of_the_audio',shade= True)
qualityPlot.set(xlim=(2.5, overview['overall_quality_of_the_audio'].max()))
qualityPlot.set_axis_labels('overall_quality_of_the_audio', 'Proportion')
qualityPlot


# In[3]:


# en_core_sci_sm = 'C:\\Users\\lalit\\Downloads\\scapy\\Scispacy Pretrained Models\\en_core_sci_sm-0.1.0\\en_core_sci_sm\\en_core_sci_sm-0.1.0'
# nlp = spacy.load(en_core_sci_sm)


# In[1]:


# !pip install -U spacy
# !pip install scispacy


# In[2]:


# !pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz


# In[ ]:


# *****************************************************************************#
# Parkinson Disease 


# In[79]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from xgboost import XGBClassifier


# In[81]:


import os
for dirname, _, filenames in os.walk('C:\\Users\\lalit\\Downloads'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Attempt 2


# In[82]:


df=pd.read_csv('C:\\Users\\lalit\\Downloads\\parkinsons.data')
print(df.shape,'\n')
df.head()


# In[83]:


X_un=df.copy()
X_un=X_un.drop(['name','status'],axis=1)
y_un=df[['status']]


# In[84]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=3,test_size=0.2,random_state=24)

for train_index,test_index in split.split(X_un,y_un):
    strat_train_set_x,strat_train_y=X_un.loc[train_index],y_un.loc[train_index]
    strat_test_set_x,strat_test_y=X_un.loc[test_index],y_un.loc[test_index] 
    


# In[85]:


from sklearn.metrics import precision_score, accuracy_score,recall_score
model=XGBClassifier(use_label_encoder=False,eval_metric='rmse')
model.fit(strat_train_set_x,strat_train_y)
y_pred=model.predict(strat_test_set_x)

print('accuracy score %.2f'% accuracy_score(y_pred,strat_test_y))
print('recall score %.2f'%recall_score(y_pred,strat_test_y))


# In[ ]:




