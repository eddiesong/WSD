from xml.dom import minidom
from sklearn import datasets
from sklearn import svm
from sklearn import neighbors
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
import json
import codecs
import sys
import unicodedata
import nltk
import math
# import wordnet

window = 10

def parse_data(input_file, language):
	'''
	Parse the .xml data file

	return a dict of list of contexts
	'''

	tokenizer = RegexpTokenizer(r'\w+')
	
	if language == 'English':
		xmldoc = minidom.parse(input_file)
		data = {}
		lex_list = xmldoc.getElementsByTagName('lexelt')
		for node in lex_list:
			lexelt = node.getAttribute('item')
			data[lexelt] = []
			inst_list = node.getElementsByTagName('instance')
			for inst in inst_list:
				instance_id = inst.getAttribute('id')
				l = inst.getElementsByTagName('context')[0]
				tokens_left = tokenizer.tokenize((l.childNodes[0].nodeValue).replace('\n', ''))[-window:]
				tokens_right = tokenizer.tokenize((l.childNodes[2].nodeValue).replace('\n', ''))[:window]
				context = tokens_left + tokens_right
				# context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')
				data[lexelt].append((instance_id, context))

	elif language == 'Spanish':
		xmldoc = minidom.parse(input_file)
		data = {}
		lex_list = xmldoc.getElementsByTagName('lexelt')
		for node in lex_list:
			lexelt = node.getAttribute('item')
			data[lexelt] = []
			inst_list = node.getElementsByTagName('instance')
			for inst in inst_list:
				instance_id = inst.getAttribute('id')
				temp = inst.getElementsByTagName('context')[0]
				l = temp.getElementsByTagName('target')[0]
				tokens_left = tokenizer.tokenize((l.childNodes[0].nodeValue).replace('\n', ''))[-window:]
				tokens_right = tokenizer.tokenize((l.childNodes[2].nodeValue).replace('\n', ''))[:window]
				context = tokens_left + tokens_right
				# context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')
				data[lexelt].append((instance_id, context))

	elif language == 'Catalan':
		xmldoc = minidom.parse(input_file)
		data = {}
		lex_list = xmldoc.getElementsByTagName('lexelt')
		for node in lex_list:
			lexelt = node.getAttribute('item')
			data[lexelt] = []
			inst_list = node.getElementsByTagName('instance')
			for inst in inst_list:
				instance_id = inst.getAttribute('id')
				temp = inst.getElementsByTagName('context')[0]
				l = temp.getElementsByTagName('target')[0]
				tokens_left = nltk.word_tokenize((l.childNodes[0].nodeValue).replace('\n', ''))[-window:]
				tokens_right = nltk.word_tokenize((l.childNodes[2].nodeValue).replace('\n', ''))[:window]
				context = tokens_left + tokens_right
				# context = (l.childNodes[0].nodeValue + l.childNodes[1].firstChild.nodeValue + l.childNodes[2].nodeValue).replace('\n', '')
				data[lexelt].append((instance_id, context))

	return data

def relevance(word, sense, data):

	sense_list = sense[word]
	contexts = data[word]

	word_set = []
	temp = []

	for sense in sense_list:
		if sense not in temp:
			temp.append(sense)

	for sense in temp:
		relevance = {}
		for instance_id, context in contexts:
			for token in context:
				i = 0
				num_sc = 0
				num_c = 0
				for instance_id, context in contexts:
					if token in context and sense_list[i] == sense:
						num_sc += 1
					if token in context:
						num_c += 1
					i += 1
				if 1 - num_sc/(num_c*1.0) == 0:
					relevance[token] = 32767
				elif num_sc/(num_c*1.0) == 0:
					relevance[token] = -32767
				else:
					relevance[token] = math.log((float)(num_sc/(num_c*1.0))/(1 - num_sc/(num_c*1.0)))
		
		for token, value in relevance.iteritems():
			if value >= 0 and token not in word_set:
				word_set.append(token)

	return word_set

def build_train_vector(data, sense, language):
	'''
	Build the context vector for each instance of a word
	'''

	if language == 'English':
		stop_words = stopwords.words('english')
		stemmer = SnowballStemmer("english", ignore_stopwords=True)
	elif language == 'Spanish':
		stop_words = stopwords.words('spanish')
		stemmer = SnowballStemmer("spanish", ignore_stopwords=True)
	else:
		stop_words = []

	vector = {}
	word_sets = {}
	for word, key in data.iteritems():
		word_set = []
		vector[word] = []
		for instance_id, context in key:
			for item in context:
				if language == 'English' or language == 'Spanish':
					after = stemmer.stem(item.lower())
				else:
					after = item.lower()
				if (after not in word_set) and (after not in stop_words):
					word_set.append(after)

		# word_set = relevance(word, sense, data)
		word_sets[word] = word_set

		for instance_id, context in key:
			context_vector = [0] * len(word_set)
			for item in context:
				if language == 'English' or language == 'Spanish':
					after = stemmer.stem(item.lower())
				else:
					after = item.lower()
				if after in word_set:
					index = word_set.index(after)
					context_vector[index] += 1
			vector[word].append(context_vector)

	return vector, word_sets

def build_dev_vector(data, word_sets, language):
	'''
	Build the context vector for each instance of a word
	'''
	vector = {}
	id_lists = {}

	if language == 'English':
		stemmer = SnowballStemmer("english", ignore_stopwords=True)
	elif language == 'Spanish':
		stemmer = SnowballStemmer("spanish", ignore_stopwords=True)

	for word, key in data.iteritems():
		vector[word] = []
		word_set = word_sets[word]

		id_list = []

		for instance_id, context in key:

			id_list.append(instance_id)

			context_vector = [0] * len(word_set)
			for item in context:
				if language == 'English' or language == 'Spanish':
					after = stemmer.stem(item.lower())
				else:
					after = item.lower()
				if after in word_set:
					index = word_set.index(after)
					context_vector[index] += 1
			vector[word].append(context_vector)

		id_lists[word] = id_list

	return vector, id_lists

def build_sense(input_file, language):
	'''
	Count the frequency of each sense
	'''

	xmldoc = minidom.parse(input_file)
	data = {}
	lex_list = xmldoc.getElementsByTagName('lexelt')
	sense_dict = []
	for node in lex_list:
		lexelt = node.getAttribute('item')
		data[lexelt] = {}
		inst_list = node.getElementsByTagName('instance')
		sense_list = []
		for inst in inst_list:
			sense_id = inst.getElementsByTagName('answer')[0].getAttribute('senseid')
			sense_list.append(sense_id)
			if sense_id not in sense_dict:
				sense_dict.append(sense_id)

		data[lexelt] = sense_list

	for lexelt, sense_list in data.iteritems():
		for i in range(len(sense_list)):
			if sense_list[i] is not "U":
				sense_list[i] = sense_dict.index(sense_list[i])
			else:
				sense_list[i] = -1

	return data, sense_dict

def SVC_predict(train_vector, sense, dev_vector, language):
	clfSVC = svm.LinearSVC()

	predict = {}

	for key, context_list in train_vector.iteritems():
		sense_list = sense[key]

		clfSVC.fit(context_list, sense_list)
		
		context_dev = dev_vector[key]
		predict_sense = clfSVC.predict(context_dev)

		predict[key] = predict_sense

	return predict

def NEI_predict(train_vector, sense, dev_vector, language):
	clfNEI = neighbors.KNeighborsClassifier()

	predict = {}

	for key, context_list in train_vector.iteritems():
		sense_list = sense[key]

		clfNEI.fit(context_list, sense_list)
		
		context_dev = dev_vector[key]
		predict_sense = clfNEI.predict(context_dev)

		predict[key] = predict_sense

	return predict

def replace_accented(input_str):
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])

def output(predict_vector, id_lists, sense_dict, output_file):
	outfile = codecs.open(output_file, encoding = 'utf-8', mode = 'w')

	for key, sense_list in predict_vector.iteritems():
		id_list = id_lists[key]
		for i in range(len(sense_list)):
			string = key + ' ' + id_list[i] + ' ' + sense_dict[sense_list[i]]
			string = replace_accented(string)
			outfile.write(string + '\n')

	outfile.close()

if __name__ == '__main__':
	if len(sys.argv) != 5:
		print 'Usage: python main.py <training_file> <test_file> <output_file> <language>'
		sys.exit(0)
	else:
		train_data = parse_data(sys.argv[1], sys.argv[4])
		sense, sense_dict = build_sense(sys.argv[1], sys.argv[4])
		train_vector, word_sets = build_train_vector(train_data, sense, sys.argv[4])
		dev_data = parse_data(sys.argv[2], sys.argv[4])
		dev_vector, id_lists = build_dev_vector(dev_data, word_sets, sys.argv[4])
		predict_vector = SVC_predict(train_vector, sense, dev_vector, sys.argv[4])
		output(predict_vector, id_lists, sense_dict, sys.argv[3])

		