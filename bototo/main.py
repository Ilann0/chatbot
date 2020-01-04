from datetime import datetime
import random
import pickle
import json
import os

from .news import get_current_news
from .swear_words import has_swear_words

import tensorflow as tf
import numpy as np
import tflearn
import nltk
from nltk.stem.lancaster import LancasterStemmer

# -------------------------------------------------------------------
# If you encounter problems in launch try this
# p.s.: you need python 3.6

# import ssl

# if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
#         getattr(ssl, '_create_unverified_context', None)):
#     ssl._create_default_https_context = ssl._create_unverified_context
#     nltk.download('punkt')



# -------------------------------------------------------------------
# DATA PRE-PROCESSING

stemmer = LancasterStemmer()

PATH = os.path.dirname(os.path.abspath(__file__))

with open(PATH + '/intents.json', 'r') as file:
    data = json.load(file)

try:
    # uncomment_this_to_retrain_the_model(there, are, two, of, me)
    with open(PATH + '/data.pickle', 'rb') as file:
        root_words, intents, training, output = pickle.load(file)
except:
    root_words = []
    intents = []
    clean_patterns = []
    patterns_matching_intent = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # split sentence to individual components
            tokenized_pattern = nltk.word_tokenize(pattern.lower())
            # strip words off of unnecessary letters (extracts word roots)
            stemed_pattern = [stemmer.stem(w)
                              for w in tokenized_pattern if w not in '?.,!']

            root_words.extend(stemed_pattern)
            clean_patterns.append(stemed_pattern)
            patterns_matching_intent.append(intent['tag'])

        if intent['tag'] not in intents:
            intents.append(intent['tag'])

    root_words = sorted(set(root_words))
    intents = sorted(intents)

    training = []
    output = []

    template_output_row = [0 for word in intents]

    # One-hot encoding data
    for idx, pattern in enumerate(clean_patterns):
        bag_of_words = []

        for word in root_words:
            if word in pattern:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)

        output_row = template_output_row.copy()
        output_row[intents.index(patterns_matching_intent[idx])] = 1

        training.append(bag_of_words)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open(PATH + '/data.pickle', 'wb') as file:
        pickle.dump((root_words, intents, training, output), file)


# -------------------------------------------------------------------
# MODEL DEFINITION

tf.reset_default_graph()

network = tflearn.input_data(shape=[None, len(root_words)])
network = tflearn.fully_connected(incoming=network, n_units=8)
network = tflearn.fully_connected(incoming=network, n_units=8)
network = tflearn.fully_connected(
    incoming=network,
    n_units=len(output[0]),
    activation='softmax')
network = tflearn.regression(incoming=network)

model = tflearn.DNN(network)


# --------------------------------------------------------------------
# MODEL TRAINING

try:
    # uncomment_this_to_retrain_the_model(there, are, two, of, me)
    model.load(PATH + '/model.tflearn')
except:
    model.fit(training, output, n_epoch=1000, batch_size=10, show_metric=True)
    model.save(PATH + '/model.tflearn')


# --------------------------------------------------------------------
# GENERAL FUNCTIONS

def input_to_one_hot(sentence, words):
    bag_of_words = [0 for _ in words]

    sentence = nltk.word_tokenize(sentence)
    sentence = [stemmer.stem(word.lower()) for word in sentence]

    for sent_word in sentence:
        for index, word in enumerate(words):
            if sent_word == word:
                bag_of_words[index] = 1

    return np.array(bag_of_words)


def predictions_to_intent(tf_prediction):
    pred_max_index = np.argmax(tf_prediction)
    pred_max_val = tf_prediction[0][pred_max_index]
    if pred_max_val > 0.70:
        intent_tag = intents[pred_max_index]
    else:
        intent_tag = "confused"

    return intent_tag

def get_time():
    time = datetime.now().time()
    
    return (f"It is currently {time.hour} hours and {time.minute} minutes")

def dispatch(user_input):
    if has_swear_words(user_input):
        return 'I am very mad at you for using that word!', 'no'
    
    converted_input = input_to_one_hot(user_input, root_words)
    predictions = model.predict([converted_input])
    
    intent_tag = predictions_to_intent(predictions)
    
    for intent in data['intents']:
        if intent_tag == intent['tag']:
            animation = random.choice(intent['animations'])
            msg = random.choice(intent['responses'])
            break
    
    if intent_tag == "news":
        msg = get_current_news()
    elif intent_tag == "time":
        msg = get_time()
    
    return msg, animation





def chat():
    print()
    print('Chat with boto:')

    while True:
        u_input = input('You: ')
        if u_input.lower() == 'quit':
            print('Ok goodbye!')
            break

        converted_input = input_to_one_hot(u_input, root_words)
        predictions = model.predict([converted_input])
        for i, prediction in enumerate(predictions[0]):
            print(intents[i] + ' >>', str((prediction * 100))[:5] + '%')

        intent_tag = predictions_to_intent(predictions)
        print()

        for intent in data['intents']:
            if intent_tag == intent['tag']:
                print('Boto:', random.choice(intent['responses']))
                break


if __name__ == '__main__':
    chat()
