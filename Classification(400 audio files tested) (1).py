#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import cmudict,stopwords
from nltk.util import ngrams
from collections import Counter
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from keras.layers import TimeDistributed, Activation, Input,Dense, Dropout, LSTM, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizer_v2.adam import Adam
import glob
from PIL import Image
from matplotlib import cm
import librosa
import csv
import os
import pathlib
import random
from scipy.io import wavfile
from tqdm import tqdm_notebook as tqdm
import tensorflow as tf
from collections import Counter


# In[2]:


df=pd.read_csv("E:\\Education\\sem 10\\overview-of-recordings.csv")
df.head()


# In[3]:


df=df[['file_name','phrase','prompt','overall_quality_of_the_audio','speaker_id']]
df.head() #using only selected columns


# In[4]:


test_num=random.randrange(0,len(df))
test_file_name=df.loc[test_num,'file_name']
print(test_file_name)
print(test_num)


# In[5]:


print(df.loc[test_num,"prompt"]+'\n'+df.loc[test_num,'phrase'])


# In[6]:


import IPython
sr=25000
display_audio_file = f"E:/Education/sem 10/recordings/test/1249120_43951421_52837630.wav"
IPython.display.Audio(display_audio_file,rate=sr)


# In[7]:


grouped_series = df.groupby('prompt').agg('count')['speaker_id'].sort_values(ascending=False)
unique_prompts = len(df['prompt'].unique())
print("Number of unique prompts : ", unique_prompts)

sns.barplot(x=grouped_series.values,y= grouped_series.index)
plt.title('Prompts to be Used as Classification Targets')
sns.despine()


# In[8]:


preprocess_df=df.drop('overall_quality_of_the_audio',axis=1)
preprocess_df


# In[9]:


fig=plt.figure(figsize=(10,4))
sns.distplot(df['overall_quality_of_the_audio'],hist=False,color='teal')
sns.despine()


# In[10]:


stop_words = set(stopwords.words('english')) 
word_dict = {}
preprocess_df['phrase'] = [i.lower() for i in preprocess_df['phrase']]
preprocess_df['phrase'] = [i.replace('can\'t', 'can not') for i in preprocess_df['phrase']]
preprocess_df['phrase'] = [i.replace('i\'m', 'i am') for i in preprocess_df['phrase']]
preprocess_df['phrase'] = [i.replace('i\'ve', 'i have') for i in preprocess_df['phrase']]
preprocess_df['phrase'] = [' '.join([j for j in i.split(' ') if j not in stop_words]) for i in preprocess_df['phrase']]

for phrase in preprocess_df['phrase']:
    for word in phrase.split(' '):
        word = word.lower()
        if word in stop_words or word == '':
            pass
        elif word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1
            
sorted_word_list = sorted(word_dict.items(), key=lambda kv: kv[1], reverse=True)


# In[11]:


# e n most common words in the phrases.
n=25

fig=plt.figure(figsize=(10,7))
plt.style.use('ggplot')
sns.barplot(x=[i[1] for i in sorted_word_list[:n]],y=[i[0] for i in sorted_word_list[:n]])


# In[12]:


base_dir = 'E:\\Education\\sem 10\\recordings - Copy\\'

train_files = [base_dir + 'train\\' + i for i in os.listdir(base_dir + 'train')]
val_files = [base_dir + 'validate/' + i for i in os.listdir(base_dir + 'validate')]
test_files = [base_dir + 'test/' + i for i in os.listdir(base_dir + 'test')]

all_files = train_files + test_files + val_files
len(all_files)


# In[13]:


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(preprocess_df['phrase'])
word_index = tokenizer.word_index
vocab_size = len(word_index)
print(f'vocab_size : {vocab_size}')

phrases_seq = tokenizer.texts_to_sequences(preprocess_df['phrase'])
padded_phrases_seq = pad_sequences(phrases_seq, padding='post')
padded_phrases_seq = np.asarray(padded_phrases_seq)
max_seq_length = padded_phrases_seq.shape[0]
print("padded_phrases_seq shape : ", padded_phrases_seq.shape)


# In[14]:


random_phrase_num = random.randrange(0, len(preprocess_df))
random_import_phrase = df.loc[random_phrase_num, 'phrase']
random_phrase = preprocess_df.loc[random_phrase_num, 'phrase']

print('padded_phrase example : ' + '\n' + random_import_phrase + '\n' + random_phrase + '\n' + str(padded_phrases_seq[random_phrase_num]))


# In[16]:


#os.listdir('../input/x-wav-array/x_wav_array.npy')
#/x_wav_array.npy')

# os.mkdir("E:\\Education\\sem 10\\recordings - Copy\\spectro2")


# In[19]:


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
    filename = Path("E:\\Education\\sem 10\\recordings - Copy\\spectro2\\" + '.jpg')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


# In[20]:


for file in tqdm(all_files, total=len(all_files)):
    create_spectrogram(file, file.split('\\')[-1])


# In[16]:


spec_filelist = [f'E:/Education/sem 10/recordings - Copy/spectro/{i}.jpg' for i in preprocess_df.file_name]
print(spec_filelist)


# In[17]:


x_wav_array = np.array([np.array(Image.open(fname)) for fname in spec_filelist])
print(x_wav_array.shape)


# In[ ]:


from numpy import save
save("x_wav_array.npy",x_wav_array)


# In[22]:


x_wav_array = np.load('x_wav_array.npy',allow_pickle=True)
print(x_wav_array.shape)


# In[23]:


print(x_wav_array.shape)


# In[17]:


enc = OneHotEncoder(handle_unknown='ignore')
prompt_array = preprocess_df['prompt'].values.reshape(-1,1)
labels_onehot = enc.fit_transform(prompt_array).toarray()

labels_onehot.shape

#exclude_file = "1249120_44142156_100535941.wav.jpg"
#spec_filelist = [
  #  f'E:/Education/sem 10/recordings - Copy/spectro/{i}.jpg'
 #   for i in preprocess_df.file_name
   # if i != exclude_file
#]
#x_wav_array = np.array([np.array(Image.open(fname)) for fname in spec_filelist])


# In[35]:


# exclude_file = "1249120_44142156_100535941.wav.jpg"
# spec_filelist = [f'E:\\Education\\sem 10\\recordings - Copy\\spectro\\{i}.jpg' for i in all_files if i != '1249120_44142156_100535941.wav.jpg']
# x_wav_array = ([np.array(Image.open(fname)) for fname in spec_filelist])
# # x_wav_array = np.load('../input/x-wav-array/x_wav_array.npy')
# # print(x_wav_array.shape)
# print(spec_filelist)
# print()
# # print(x_wav_array)


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(preprocess_df.index, labels_onehot, test_size = .2)


# In[24]:


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


# In[25]:


print(x_wav_train.shape)


# In[3]:


import pandas as pd


# In[7]:


#!pip install fastai


# In[2]:


#https://www.kaggle.com/code/paultimothymooney/medical-symptoms-text-and-audio-classification
from fastai.text import *
from fastai.vision import *
import spacy
from spacy import displacy
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


# In[5]:


#!pip install scispacy
#import scispacy


# In[6]:


import scispacy


# In[8]:


Data_dir_train=np.array(glob("E://Education//sem 10//recordings - Copy//train//*"))
Data_dir_test=np.array(glob("E://Education//sem 10//recordings - Copy//test//*"))
Data_dir_val=np.array(glob("E://Education//sem 10//recordings - Copy//validate//*"))

for file in tqdm(Data_dir_train):
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_melspectrogram(filename,name)
for file in tqdm(Data_dir_test):
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_melspectrogram(filename,name)
for file in tqdm(Data_dir_val):
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_melspectrogram(filename,name)


# In[28]:


tfms = fastai.get_transforms()


# In[30]:


from pathlib import Path
path = Path('E:\\Education\\sem 10\\recordings - Copy\\')
np.random.seed(7)
data = ImageDataLoaders.from_df(path,df="train", folder="spectro", valid_pct=0.2, suffix='.jpg',
        ds_tfms=aug_transforms(max_rotate=25), size=299, num_workers=0).normalize(imagenet_stats)
#get_transforms()
learn = create_cnn(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(10)


# In[15]:


from fastai import ImageDataLoaders


# In[16]:


from fastai.vision.all import *


# In[20]:


from fastai import *
from fastai.vision import *


# In[35]:


OUTPUT_DIR="E:\\Education\\sem 10\\recordings - Copy\\output\\"
audio_images="E:\\Education\\sem 10\\recordings - Copy\\spectro\\"


# In[36]:


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 25

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=os.path.join(OUTPUT_DIR,audio_images),
                                             shuffle=True,
                                             color_mode='rgb',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="training",
                                             seed=0)


# In[3]:


INPUT_DIR="E:\\Education\\sem 10\\Medical\\train\\"
OUTPUT_DIR="E:\Education\sem 10\Medical\\"


# In[38]:


parent_list = os.listdir(INPUT_DIR)
for i in range(10):
    print(parent_list[i])


# In[4]:


import tensorflow as tf
import matplotlib.pyplot as plt
import os
import wave
import pylab
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools


# In[41]:


# Plot first 5 WAV files as a waveform and a frequency spectrum
for i in range(5): 
    signal_wave = wave.open(os.path.join(INPUT_DIR, parent_list[i]), 'r')
    sample_rate = 16000
    sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)

    plt.figure(figsize=(12,12))
    plot_a = plt.subplot(211)
    plot_a.set_title(parent_list[i])
    plot_a.plot(sig)
    plot_a.set_xlabel('sample rate * time')
    plot_a.set_ylabel('energy')

    plot_b = plt.subplot(212)
    plot_b.specgram(sig, NFFT=1024, Fs=sample_rate, noverlap=900)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')

plt.show()


# In[7]:


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

# For every recording, make a spectogram and save it as label_speaker_no.png
if not os.path.exists(os.path.join(OUTPUT_DIR, 'audio-images')):
    os.mkdir(os.path.join(OUTPUT_DIR, 'audio-images'))
    
for filename in os.listdir(INPUT_DIR):
    if "wav" in filename:
        file_path = os.path.join(INPUT_DIR, filename)
        file_stem = Path(file_path).stem
        target_dir = f'class_{file_stem[0]}'
        dist_dir = os.path.join(os.path.join(OUTPUT_DIR, 'audio-images'), target_dir)
        file_dist_path = os.path.join(dist_dir, file_stem)
        if not os.path.exists(file_dist_path + '.png'):
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)
            file_stem = Path(file_path).stem
            sound_info, frame_rate = get_wav_info(file_path)
            pylab.specgram(sound_info, Fs=frame_rate)
            pylab.savefig(f'{file_dist_path}.png')
            pylab.close()

# Print the ten classes in our dataset
path_list = os.listdir(os.path.join(OUTPUT_DIR, 'audio-images'))
print("Classes: \n")
for i in range(3):
    print(path_list[i])
    
# File names for class 1
path_list = os.listdir(os.path.join(OUTPUT_DIR, 'audio-images/class_1'))
print("\nA few example files: \n")
for i in range(10):
    print(path_list[i])


# In[8]:


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 32
N_CHANNELS = 3
N_CLASSES = 10

# Make a dataset containing the training spectrograms
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                             batch_size=BATCH_SIZE,
                                             validation_split=0.2,
                                             directory=os.path.join(OUTPUT_DIR, 'audio-images'),
                                             shuffle=True,
                                             color_mode='rgb',
                                             image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                             subset="training",
                                             seed=0)


# In[9]:


plt.figure(figsize=(12, 12))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()


# In[10]:


def prepare(ds, augment=False):
    # Define our one transformation
    rescale = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.Rescaling(1./255)])
    flip_and_rotate = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
    ])
    
    # Apply rescale to both datasets and augmentation only to training
    ds = ds.map(lambda x, y: (rescale(x, training=True), y))
    if augment: ds = ds.map(lambda x, y: (flip_and_rotate(x, training=True), y))
    return ds

train_dataset = prepare(train_dataset, augment=False)
#valid_dataset = prepare(valid_dataset, augment=False)


# In[11]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS)))
model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))


# In[12]:


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],
)

# Train model for 10 epochs, capture the history
history = model.fit(train_dataset, epochs=10)  # we got an accuracy of 98 for 380 audio files


# In[14]:


history_dict = history.history
loss_values = history_dict['loss']
#val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)

plt.figure(figsize=(8,6))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
#plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[15]:


# Plot the accuracy curves for training and validation.
acc_values = history_dict['accuracy']
#val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(acc_values)+1)

plt.figure(figsize=(8,6))
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
#plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:




