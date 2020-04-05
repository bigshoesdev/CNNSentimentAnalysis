from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import collections
import re
import string
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer


def remove_punct(text):
    text_nopunct = ''
    text_nopunct = re.sub('[' + string.punctuation + ']', '', text)
    return text_nopunct


def lower_token(tokens):
    return [w.lower() for w in tokens]


def remove_stop_words(tokens):
    return [word for word in tokens if word not in stoplist]


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors,
                                                                               generate_missing=generate_missing))
    return list(embeddings)


data = pd.read_csv('imdb_labelled.tsv', header=None, delimiter='\t')

data.columns = ['Text', 'Label']

print(data.head())

pos = []
neg = []
for l in data.Label:
    if l == 0:
        pos.append(0)
        neg.append(1)
    elif l == 1:
        pos.append(1)
        neg.append(0)

data['Pos'] = pos
data['Neg'] = neg

print(data.head())

data['Text_Clean'] = data['Text'].apply(lambda x: remove_punct(x))
tokens = [word_tokenize(sen) for sen in data.Text_Clean]

lower_tokens = [lower_token(token) for token in tokens]
stoplist = stopwords.words('english')
filtered_words = [remove_stop_words(sen) for sen in lower_tokens]
result = [' '.join(sen) for sen in filtered_words]
data['Text_Final'] = result

data['tokens'] = filtered_words
data = data[['Text_Final', 'tokens', 'Label', 'Pos', 'Neg']]
data[:4]

data_train, data_test = train_test_split(data, test_size=0.10, random_state=42)

all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 300

tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(data_train["Text_Final"].tolist())
training_sequences = tokenizer.texts_to_sequences(data_train["Text_Final"].tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))
print('===================training_sequences==============================')
print(training_sequences[:2])

train_cnn_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('===================train_cnn_data==============================')
print(train_cnn_data[:2])

word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

train_embedding_weights = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))
for word, index in train_word_index.items():
    train_embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print('===================train_embedding_weights.shape==============================')
print(train_embedding_weights.shape)

print('===================train_embedding_weights==============================')
print(train_embedding_weights)

test_sequences = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=False)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = [2, 3, 4, 5, 6]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.1)(l_merge)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    preds = Dense(labels_index, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model


label_names = ['Pos', 'Neg']
y_train = data_train[label_names].values
x_train = train_cnn_data
y_tr = y_train
model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index) + 1, EMBEDDING_DIM,
                len(list(label_names)))

num_epochs = 3
batch_size = 34

hist = model.fit(x_train, y_tr, epochs=num_epochs, validation_split=0.1, shuffle=True, batch_size=batch_size)

predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)

labels = [1, 0]
prediction_labels = []
for p in predictions:
    prediction_labels.append(labels[np.argmax(p)])

print(sum(data_test.Label == prediction_labels) / len(prediction_labels))


print(data_test.Label.value_counts())


if __name__ == '__main__':
    pass