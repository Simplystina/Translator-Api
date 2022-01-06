import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from keras.preprocessing.sequence import pad_sequences
import pickle
import string

#model = tf.keras.models.load_model(r'C:\Users\user\Documents\saved_model\my_model\saved_model.pb' )


cwd = os.getcwd()
model = tf.keras.models.load_model(os.path.join(cwd+"\saved_model\my_model"))


file = cwd+"\\tokenizer.pickle"

with open(f"{cwd}\\tokenizer.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)


file = cwd+"\pidgin_tokenizer.pickle"

with open(file, 'rb') as handle:
    pidgin_tokenizer = pickle.load(handle)


file = cwd+"\dict.pickle"

with open(file, 'rb') as handle:
    dict_data = pickle.load(handle)




array  = np.load(cwd+r"\x_data.npy")


pidgin = np.load(cwd+"\pidign_array.npy")



sentence = 'eternal rock of ages'

def preprocess(words):
    result = ''.join([i for i in words if not i.isdigit()])
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in result if ch not in exclude)
    return s.lower()


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def predict(sentence):
    sentence = preprocess(sentence)
    #sentence = [tokenizer.word_index[word]if True else 0 for word in sentence.split()]
    sentences =[]
    for word in sentence.split():
        try:
            sentences.append(tokenizer.word_index[word])
        except KeyError:
            continue
    sentence = sentences
    sentence = pad_sequences([sentence], maxlen=array.shape[-1], padding='post')
    sentences = np.array([sentence[0], array[0]])
    predictions = model.predict(sentences, len(sentences))
    print(predictions)
    #print('Sample 1:')
    predd = ' '.join([dict_data[np.argmax(x)] for x in predictions[0]])

    #print('Sample 2:')
    print(' '.join([dict_data[np.argmax(x)] for x in predictions[0]]))
    pred = ' '.join([dict_data[np.max(x)] for x in pidgin[1]])
    return pred.split('<PAD>')[0]
    #return ' '.join([dict_data[np.max(x)] for x in pidgin[0]])

predict(sentence)


app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def call_sentence():
    # Catch the text from a POST request
    if request.method == 'POST':
        # Return on a JSON format
        return jsonify(predict(sentence))
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
