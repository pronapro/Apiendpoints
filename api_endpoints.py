from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request,jsonify
import pickle
import keras
from keras.preprocessing.sequence import pad_sequences
import time
import flask
import numpy as np
from summarize_nltk import nltk_summarizer
import time

app = Flask(__name__)
#Let's load the best model obtained during training
best_model = keras.models.load_model("best_model1.hdf5")
max_words = 5000
max_len = 200
with open('tokenizer.pickle', 'rb') as handle:

	tokenizer = pickle.load(handle)

# loading tokenizer
def preprocess_texts(text):
	max_words = 5000
	max_len = 200
    # saving tokenizer
	with open('tokenizer.pickle', 'rb') as handle:

		tokenizer = pickle.load(handle)
	sequence = tokenizer.texts_to_sequences(text)
	test = pad_sequences(sequence, maxlen=max_len)
	return test


@app.route('/index')
def first():
	return "ok"

@app.route('/api/sentiment_analysis', methods =['GET','POST'])
def lstm_sentiment():
	if request.method == 'POST':
		text = request.form['text']
		#array of sentiment
		sentiment = ['Neutral','Negative','Positive']
		sequence = tokenizer.texts_to_sequences([text])
		test = pad_sequences(sequence, maxlen=max_len)
		senti = sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]
		"""
		text = request.form['text']
		text = preprocess_texts(text)
		print(text)
		#senti = sentiment[np.around(best_model.predict(text), decimals=0).argmax(axis=1)[0]]
		senti = sentiment[np.around(best_model.predict(text), decimals=0).argmax(axis=1)[0]]"""
		return(jsonify(sentiment =senti))

@app.route('/api/text_summarization',methods=['GET','POST'])
def text_sum():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		#final_reading_time = readingTime(rawtext)
		final_summary_nltk = nltk_summarizer(rawtext)
		#summary_reading_time_nltk = readingTime(final_summary_nltk)

		end = time.time()
		final_time = end-start
		print(final_time)
	return flask.jsonify(Summary=final_summary_nltk , Time=final_time)




if __name__ == '__main__':
	app.run(debug=True)