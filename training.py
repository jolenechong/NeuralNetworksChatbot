import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
# reduce word to its stem, looks for the exact word instead -> work working worked works will change to work

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

# lematise individual words
# we will be using document-term matrix formatting
lemmatizer = WordNetLemmatizer()

# open and read json file
intents = json.loads(open('intents.json').read())

words = []
# for tags
classes = []
documents = []
ignore_letters = ['?','!','.',',']

for intent in intents['intents']:
    # intents is the object in json and we need to access the key intents
    for pattern in intent["patterns"]:
        # for each in patterns list in json
        word_list = nltk.word_tokenize(pattern)
        # above -> splits into individual words
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        # keep word list and the type tgt
        if intent['tag'] not in classes:
            # if tag not alr in classes, add it to classes
            classes.append(intent['tag'])

# print(documents)


# lemmatize the words in word list if its not part of the ignore_letters list
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# eliminate duplicates using set (cleaning the data)
words = sorted(set(words))

# there shouldnt be duplicates, but just clear duplicates
classes = sorted(set(classes))

# save files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))




# MACHINE LEARNING - NEURAL NETWORKS
# representing words as numerical values

# make a bag of words
training = []
output_empty = [0]*len(classes)

# print(documents)

# to explain the part below, heres a sample list of the output of documents
# [(['hello'], 'greetings'),
# (['hey'], 'greetings'),
# (['hi'], 'greetings'),
# (['good', 'day'], 'greetings'),
#  (['whatsup'], 'greetings'),
# (['how', 'is', 'it', 'going'], 'greetings'),
# (['wassup'], 'greetings'),
# (['bye'], 'goodbye')]
# this means "document" in the list below is (['hello'],'greetings') for eg
# document[0] means taking hello


for document in documents:
    bag = []
    word_patterns = document[0]
    # example output for the thing below this comment
    # ['how', 'is', 'it', 'going']
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

# creating sequential model here
model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# in is the learning rate
sgd = SGD(lr=0.01,decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x),np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5',hist)
print('done')
