import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, SpatialDropout1D
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
import streamlit as st

dataset_izzan_1 = pd.read_csv('train.csv')
dataset_izzan_2 = pd.read_csv('test.csv')

dataset_izzan_1.head()
dataset_izzan_2.head()

dataset = pd.concat([dataset_izzan_1, dataset_izzan_2], ignore_index=True)

text = dataset['original_text']

import string
import re

def cleanse(data):
    # Lowercase
    data = data.lower()

    # Remove newline
    data = data.replace('\n', ' ')

    # Remove links
    data = re.sub(r'http\S+', '', data)
    data = re.sub(r'bit.ly\S+', '', data)
    data = re.sub(r't.co\S+', '', data)
    data = re.sub(r's.id\S+', '', data)

    # Remove mentions
    data = data.replace("_", "")
    data = re.sub("(@[A-Za-z0-9]+)", " ", data)

    # Remove punctuations
    punct = string.punctuation.replace("<", "").replace(">", "").replace("[", "").replace("]", "")
    translator = str.maketrans(punct, ' ' * len(punct))
    data = data.translate(translator)

    # Remove textual emoji
    data = re.sub(r'<\s?(.*?)\s?>', r'{<\1>}', data)

    # Remove characters between [], <>, or {}
    data = re.sub("[\[].*?[\]]", " ", data)
    data = re.sub(r'<[^>]+>', ' ', data)
    data = re.sub(r'{[^>]+}', ' ', data)

    # Remove numeric digits
    data = re.sub(r'[0-9]', '', data)

    # Strip excess characters on both ends
    data_list = [re.sub(r'([a-z])\1+$', r'\1', token) for token in data.split()]
    data_list = [re.sub(r'^([a-z])\1+', r'\1', token) for token in data_list]
    data = " ".join(data_list)

    # Remove extra space
    data = ' '.join(data.split())

    return data


text_cleaned = [cleanse(sentence) for sentence in text]

dataset['original_text'] = text_cleaned

# Data frame Anda (df) dan operasi yang Anda lakukan
ax = (dataset[['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']]
      .sum(axis=0)
      .sort_values()
      .plot(kind='barh', color='darkred'))

# Menambahkan label ke setiap batang
for bar in ax.patches:
    width = bar.get_width()
    ax.annotate(f'{width}', xy=(width, bar.get_y() + bar.get_height() / 2), va='center')

plt.show()


pencemaran_nama_baik = dataset[(dataset['pencemaran_nama_baik'] == 1) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 0)]
sara = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 1) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 0)]
radikalisme = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 1) & (dataset['pornografi'] == 0)]
pornografi = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 1)]
dataset = pd.concat([pencemaran_nama_baik,sara,radikalisme,pornografi],ignore_index=True)

dataset

ax = (dataset[['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']]
      .sum(axis=0)
      .sort_values()
      .plot(kind='barh', color='darkred'))

# Menambahkan label ke setiap batang
for bar in ax.patches:
    width = bar.get_width()
    ax.annotate(f'{width}', xy=(width, bar.get_y() + bar.get_height() / 2), va='center')

plt.show()

pencemaran_nama_baik = dataset[(dataset['pencemaran_nama_baik'] == 1) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 0)].head(166)
sara = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 1) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 0)]
radikalisme = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 1) & (dataset['pornografi'] == 0)].head(166)
pornografi = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 1)].head(166)
dataset_under = pd.concat([pencemaran_nama_baik,sara,radikalisme,pornografi],ignore_index=True)

dataset_under


ax = (dataset_under[['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']]
      .sum(axis=0)
      .sort_values()
      .plot(kind='barh', color='darkred'))

# Menambahkan label ke setiap batang
for bar in ax.patches:
    width = bar.get_width()
    ax.annotate(f'{width}', xy=(width, bar.get_y() + bar.get_height() / 2), va='center')

plt.show()


pencemaran_nama_baik = dataset[(dataset['pencemaran_nama_baik'] == 1) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 0)].head(500)
sara = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 1) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 0)]
sara = pd.concat([sara,sara,sara,sara.head(2)],ignore_index=True)
radikalisme = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 1) & (dataset['pornografi'] == 0)]
radikalisme = pd.concat([radikalisme,radikalisme.head(34)],ignore_index=True)
pornografi = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 1)].head(500)
dataset_mix = pd.concat([pencemaran_nama_baik,sara,radikalisme,pornografi],ignore_index=True)

dataset_mix

ax = (dataset_mix[['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']]
      .sum(axis=0)
      .sort_values()
      .plot(kind='barh', color='darkred'))

# Menambahkan label ke setiap batang
for bar in ax.patches:
    width = bar.get_width()
    ax.annotate(f'{width}', xy=(width, bar.get_y() + bar.get_height() / 2), va='center')

plt.show()


pencemaran_nama_baik = dataset[(dataset['pencemaran_nama_baik'] == 1) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 0)]
pencemaran_nama_baik = pd.concat([pencemaran_nama_baik,pencemaran_nama_baik.head(345)],ignore_index=True)
sara = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 1) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 0)]
sara = pd.concat([sara,sara,sara,sara,sara,sara,sara,sara.head(38)],ignore_index=True)
radikalisme = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 1) & (dataset['pornografi'] == 0)]
radikalisme = pd.concat([radikalisme,radikalisme,radikalisme.head(268)],ignore_index=True)
pornografi = dataset[(dataset['pencemaran_nama_baik'] == 0) & (dataset['sara'] == 0) & (dataset['radikalisme'] == 0) & (dataset['pornografi'] == 1)].head(1200)
dataset_over = pd.concat([pencemaran_nama_baik,sara,radikalisme,pornografi],ignore_index=True)

dataset_over

ax = (dataset_over[['pornografi', 'sara', 'radikalisme', 'pencemaran_nama_baik']]
      .sum(axis=0)
      .sort_values()
      .plot(kind='barh', color='darkred'))

# Menambahkan label ke setiap batang
for bar in ax.patches:
    width = bar.get_width()
    ax.annotate(f'{width}', xy=(width, bar.get_y() + bar.get_height() / 2), va='center')

plt.show()

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.75 and logs.get('accuracy')>0.75):
      print("\nAkurasi telah mencapai >75%!")
      self.model.stop_training = True
callbacks = myCallback()




# from keras.models import Sequential
# from keras.layers import LSTM, Embedding, Dense, SpatialDropout1D
# import tensorflow as tf
# from keras.optimizers import Adam
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MultiLabelBinarizer
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping

data = [dataset,dataset_under,dataset_mix,dataset_over]

max_words = 5000
max_len = 100

for df in data:
    y_bin = df[['pencemaran_nama_baik', 'radikalisme', 'sara', 'pornografi']].values
# Tokenization and Padding
    tokenizer = Tokenizer(num_words=max_words, split=' ')
    tokenizer.fit_on_texts(df['original_text'])
    X_seq = tokenizer.texts_to_sequences(df['original_text'])
    X_pad = pad_sequences(X_seq, maxlen=max_len)

    # MultiLabelBinarizer
#     mlb = MultiLabelBinarizer()
#     y_bin = mlb.fit_transform(df['labels'])

    # Train-test split
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(X_pad, y_bin, test_size=0.2,stratify=y_bin,random_state=123)


    # Model Definition
#     model = Sequential()
#     model.add(Embedding(max_words, 128, input_length=max_len))
#     model.add(SpatialDropout1D(0.2))
#     model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, name = "LSTM_layer"))
#     num_classes = len(mlb.classes_)
#     model.add(Dense(num_classes, activation='sigmoid'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=6000, output_dim=16),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='sigmoid')
        # tf.keras.layers.Dropout(0.5)
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    # Model Training
    history = model.fit(X_train, y_train_bin, epochs=50, batch_size=32, validation_data=(X_test, y_test_bin),verbose=2)

    # Plot Loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Model Evaluation
    loss, accuracy = model.evaluate(X_test, y_test_bin)
    print("Loss:", loss)
    print("Accuracy:", accuracy)


text_input = "halo cina islam jelek banget gausah ikut campur"

# Tokenisasi teks input
input_seq = tokenizer.texts_to_sequences([text_input])
input_pad = pad_sequences(input_seq, maxlen=max_len)

# Prediksi menggunakan model yang telah dilatih
threshold = 0.5  # Misalnya, gunakan threshold 0.5

# Prediksi menggunakan model yang telah dilatih
predictions = model.predict(input_pad)

# Mendapatkan nama label
predicted_labels = ['pencemaran_nama_baik', 'radikalisme', 'sara', 'pornografi']

# Memilih label dengan nilai prediksi tertinggi
max_index = predictions[0].argmax()
max_prediction = predictions[0][max_index]
predicted_label = predicted_labels[max_index]

if max_prediction < threshold:
    predicted_label = "Tidak Tahu Kok Tanya guehh"
    
print("presentase: ",max_prediction)
print("Label yang diprediksi: ",predicted_label)