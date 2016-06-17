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
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
import time
import datetime

PATH = "./thrones/train.txt"
text = open(PATH).read()
OUTPUT_COUNT = 10000
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
batch_size = 128
if (sys.argv[1] == "play"):
    print('original corpus length:', len(text))
    batch_size = 1
    text = text[:5000]
    OUTPUT_COUNT = 100
    maxlen = 20

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
if(sys.argv[2]!="None"):
    model.load_weights(sys.argv[2])
    print('Loaded previous weights')

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=batch_size, nb_epoch=1, show_accuracy=True)

    start_index = random.randint(0, len(text) - maxlen - 1)

    to_save = []
    probabilities = [0.2, 0.5, 1.0, 1.2]
    for diversity in probabilities:
        print()
        print('----- diversity:', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')

        for i in range(OUTPUT_COUNT):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

        to_save.append(generated)
        if (sys.argv[1] == "play"):
            print(generated)

    # saving weights generated text to file
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    output_path = "./Gen_Txt/" + str(sys.argv[1]) + "/" + st + "/"
    os.makedirs(output_path)
    model.save_weights(output_path+"weights", overwrite=True)

    for index in range(len(to_save)):
        text_file = open(output_path + str(iteration) + "Iter" + str(probabilities[index]) + "Prob.txt", "w")
        text_file.write(to_save[index])
        text_file.close()
