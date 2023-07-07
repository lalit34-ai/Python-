#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import speech_recognition as sr


# In[ ]:


CREATE_CSV_FILES = True


# In[ ]:


TRAIN_CSV_FILE = "train.csv"
TEST_CSV_FILE = "test.csv"
Valdiate_CSV_FILE = "validate.csv"


# In[10]:


def extractWavFeatures(soundFilesFolder, csvFileName):
    print("The features of the files in the folder "+soundFilesFolder+" will be saved to "+csvFileName)
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    print('CSV Header: ', header)
    file = open(csvFileName, 'w', newline='')
    #with file:
    writer = csv.writer(file)
    writer.writerow(header)
    genres = '1 2 3 4 5 6 7 8 9 0'.split()
    for filename in os.listdir(soundFilesFolder):
        number = f'{soundFilesFolder}/{filename}'
        y, sr = librosa.load(number, mono=True, duration=30)
        # remove leading and trailing silence
        y, index = librosa.effects.trim(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        writer.writerow(to_append.split())
    file.close()
    print("End of extractWavFeatures")

if (CREATE_CSV_FILES == True):
    extractWavFeatures("E:\\Education\\sem 10\\recordings - Copy\\t", TRAIN_CSV_FILE) # train folder
    extractWavFeatures("E:\\Education\\sem 10\\recordings - Copy\\test", TEST_CSV_FILE)
    #extractWavFeatures("E:\Education\sem 10\recordings - Copy\\validate", Valdiate_CSV_FILE)
    print("CSV files are created")
else:
    print("CSV files creation is skipped")


# In[11]:


from sklearn import preprocessing

def preProcessData(csvFileName):
    print(csvFileName+ " will be preprocessed")
    data = pd.read_csv(csvFileName)
    # we have six speakers: 
    # 0: Jackson
    # 1: Nicolas 
    # 2: Theo
    # 3: Ankur
    # 4: Caroline
    # 5: Rodolfo
    filenameArray = data['filename'] 
    speakerArray = []
    #print(filenameArray)
    for i in range(len(filenameArray)):
        speaker = filenameArray[i][2]
        #print(speaker)
        if speaker == "j":
            speaker = "0"
        elif speaker == "n":
            speaker = "1"
        elif speaker == "t":
            speaker = "2"
        elif speaker == "a":
            speaker = "3"
        elif speaker == "c":
            speaker = "4"
        elif speaker == "r":
            speaker = "5"
        else: 
            speaker = "6"
        #print(speaker)
        speakerArray.append(speaker)
    data['number'] = speakerArray
    #Dropping unnecessary columns
    data = data.drop(['filename'],axis=1)
    data = data.drop(['label'],axis=1)
    data = data.drop(['chroma_stft'],axis=1)
    data.shape

    print("Preprocessing is finished")
    print(data.head())
    return data

trainData = preProcessData(TRAIN_CSV_FILE)
testData = preProcessData(TEST_CSV_FILE)


# In[12]:


from sklearn.model_selection import train_test_split
X = np.array(trainData.iloc[:, :-1], dtype = float)
y = trainData.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)


X_test = np.array(testData.iloc[:, :-1], dtype = float)
y_test = testData.iloc[:, -1]

print("Y from training data:", y_train.shape)
print("Y from validation data:", y_val.shape)
print("Y from test data:", y_test.shape)


# In[13]:


#Normalizing the dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()
X_train = scaler.fit_transform( X_train )
X_val = scaler.transform( X_val )
X_test = scaler.transform( X_test )

print("X from training data", X_train.shape)
print("X from validation data", X_val.shape)
print("X from test data", X_test.shape)


# In[7]:


# !pip install tensorflow


# In[14]:


from keras import models
from keras import layers
import keras


# In[16]:


#Creating a Model


# model 1
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Learning Process of a model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# simple early stopping
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)


# In[21]:


trainData.head()


# In[20]:


trainData["number"] = trainData.number.astype(float)


# In[22]:


trainData.dtypes


# In[24]:


trainData.shape


# In[8]:


#medical Utterance and intent dataset
#Train with early stopping to avoid overfitting
# history = model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=50,batch_size=128,callbacks=[es])


# In[3]:


import wave, os

path = 'E:\\Education\\sem 10\\recordings\\test'
def load_audio_file(path):
    zero = []
    for filename in glob.glob(os.path.join(path, '*.wav')):
        w = wave.open(filename, 'r')
        d = w.readframes(w.getnframes())
        zero.append(d)
        w.close()
    return zero


# In[5]:


# load_audio_file(path)


# In[6]:


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


# In[ ]:


wav_list = []
import librosa.display
from pathlib import Path
spec_dir = base_dir + 'spectrograms/'

def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[2,2])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = Path("E:\\Education\\sem 10\\recordings - Copy\\spectro" + '.jpg')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


# In[55]:


for file in tqdm(all_files, total=len(all_files)):
    create_spectrogram(file, file.split('/')[-1])


# In[35]:


exclude_file = "1249120_44142156_100535941.wav.jpg.jpg"
spec_filelist = [f'{i}.jpg' for i in all_files if i != '1249120_44142156_100535941.wav.jpg.jpg']
x_wav_array = ([np.array(Image.open(fname)) for fname in spec_filelist])
# x_wav_array = np.load('../input/x-wav-array/x_wav_array.npy')
# print(x_wav_array.shape)
print(spec_filelist)
print()
# print(x_wav_array)


# In[ ]:


enc = OneHotEncoder(handle_unknown='ignore')
prompt_array = preprocess_df['prompt'].values.reshape(-1,1)
labels_onehot = enc.fit_transform(prompt_array).toarray()

labels_onehot.shape


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(preprocess_df.index, labels_onehot, test_size = .2)


# In[ ]:


x_phrase_train = padded_phrases_seq[x_train]
x_phrase_test = padded_phrases_seq[x_test]

x_wav_train = x_wav_array[x_train]
x_wav_test = x_wav_array[x_test]

try:
    del x_wav_array
except:
    pass

x_wav_train = np.stack(x_wav_train, axis=0)
x_wav_test = np.stack(x_wav_test, axis=0)

print(x_phrase_train.shape)
print(x_phrase_test.shape)

print(x_wav_train.shape)
print(x_wav_test.shape)


# In[ ]:


def build_phrase_model(vocab_size, embedding_dim, rnn_units, max_seq_length):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size + 1, ### Without +1, layer expects [0,1160) and our onehot encoded values include 1160
                                        embedding_dim, ### Output layer size
                                        input_length =  14))
    model.add(tf.keras.layers.LSTM(rnn_units))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(unique_prompts, activation='softmax'))
    return model

model = build_phrase_model(
    vocab_size = vocab_size,
    embedding_dim=100,
    rnn_units=150,
    max_seq_length=max_seq_length)

adam_opt = Adam(lr=0.01)

model.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=2, min_delta=.005)


def exp_decay(epoch):
    initial_lrate = 0.01
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate
lrate = LearningRateScheduler(exp_decay)

callbacks_list = [earlystop_callback, lrate]

history = model.fit(x_phrase_train, y_train,
                    epochs=15, batch_size=30, validation_split = .2,
                    callbacks=callbacks_list)


# In[ ]:


model_cm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(model.predict(x_phrase_test),axis=1))

fig = plt.figure(figsize=(15,10))
sns.heatmap(model_cm, annot=True, xticklabels=enc.categories_[0].tolist(), yticklabels=enc.categories_[0].tolist())


# In[ ]:


from keras.constraints import max_norm
def build_wav_model(filters, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters, 2, 2, activation='relu', padding="same", input_shape=input_shape, kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(int(filters / 2), 2, 2, activation='relu', padding="same"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(.2))
    #model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(unique_prompts, activation='softmax'))
    return model

wav_model = build_wav_model(
    filters = 32,
    input_shape = x_wav_train[0].shape)

adam_opt = Adam(lr=0.001)

wav_model.compile(optimizer=adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])
wav_model.summary()


# In[ ]:


earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, min_delta=.005)

callbacks_list = [earlystop_callback]


# In[ ]:


history = wav_model.fit(x_wav_train, y_train,epochs=15, batch_size=20, validation_split = .2,callbacks=callbacks_list)


# In[ ]:


def alexnet(in_shape=x_wav_train[0].shape, n_classes=unique_prompts, opt='sgd'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(in_shape))
    model.add(tf.keras.layers.Conv2D(96,11, strides=4, activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(3, 2))
    model.add(tf.keras.layers.Conv2D(256,5, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(3, 2))
    model.add(tf.keras.layers.Conv2D(384, 3, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(3, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    return model


# In[ ]:


earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, min_delta=.005)

callbacks_list = [earlystop_callback]

alexnet_model = alexnet()

alexnet_model.compile(loss="categorical_crossentropy", optimizer='adam',
	              metrics=["accuracy"])


# In[ ]:


history = alexnet_model.fit(x_wav_train, y_train,epochs=15, batch_size=20, validation_split = .2,callbacks=callbacks_list)


# In[2]:


# 25.04.23
import vaex


# In[3]:


df_vaex=vaex.open("C:\\Users\\lalit\\Downloads\\mfcc_features.csv")
df_vaex


# In[1]:


# 11. 06.23
import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


df=pd.read_csv("E:\\Education\\sem 10\\overview-of-recordings.csv")
df.head()


# In[3]:


def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# In[4]:


from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn


def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1


# In[6]:


import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Speak Anything :")
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said : {}".format(text))
    except:
        print("Sorry could not recognize what you said")


# In[7]:


from nltk import tokenize
from operator import itemgetter
import math


# In[8]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words=set(stopwords.words('english'))


# In[11]:


import spacy


# In[14]:


import spacy.cli
spacy.cli.download("en_core_web_lg")


# In[15]:


get_ipython().system('pip install en_core_sci_lg')


# In[16]:


nlp = spacy.load("en_core_sci_lg")
doc=nlp(text)
keywords = []
for ent in doc.ents:
    if ent.label_ in ["NOUN", "ADJ"]:
        keywords.append(ent.text)

# Print the extracted keywords
print(keywords)


# In[10]:


get_ipython().system('pip install multi_rake')
#from multi_rake import Rake


# In[ ]:


tf_score={}
for each_word in 

