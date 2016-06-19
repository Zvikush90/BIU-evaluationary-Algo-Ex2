'''
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import numpy as np
import math
import random
import sys
import os
import time
import datetime

TEST_PATH = "./thrones/test.txt"
PATH = "./thrones/train.txt"
text = open(PATH).read()
test_text = open(PATH).read()
OUTPUT_COUNT = 10000
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
batch_size = 128

print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


def get_weights():
    weights = []
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk("./Gen_Txt/run"):
        path = root.split('/')
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            print(len(path) * '---', file)
            if(str(file)=="weights"):
                weights.append(root+"\\"+str(file))
    return weights


weights = get_weights()

for w in weights:
    print('-' * 50)
    print('Weight', w)
    model.load_weights(w)
    print('Loaded weights')
    probabilities = [0.2, 0.5, 1.0, 1.2]

    for diversity in probabilities:
        print('-' * 10)
        print('Diversity', diversity)
        success_count = 0
        count = 0
        cross_entropy = 0.0

        for c in test_text:
            preds = model.predict(c, verbose=1)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            cross_entropy = cross_entropy - math.log(preds[next_char],2)
            if (count < len(test_text)):
                if (next_char == test_text[count + 1]):
                    success_count = success_count + 1

            count = count + 1
        print('Accuracy',float(success_count/count))
        print('Cross Entropy',cross_entropy)