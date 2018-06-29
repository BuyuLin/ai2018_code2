import numpy as np
from collection import Counter
from math import *

def make_dictionary(train_dir, number):
	emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
	all_words = []
	for mail in emails:
		with open(mail) as m:
			for line in m:
				words = line.split()
				all_words += words

	dictionary  = Counter(all_words)

	list_to_remove = dictionary.keys()

	for item in list_to_remove:
		if item.isalpha() == False and item.isnumeric()==False and item[0:2] != "__":
			del dictionary[item]
		elif len(item) <= 1 and item!=  "i":
			del dictionary[item]

	dictionary = dictionary.most_common(number)
	return dictionary

def feature_extraction(mail_dir, number, dictinary):
	files = [os.path.join(mail_dir, f) for f in os.listdir(mail_dir)]
	feature_matrix = np.zeros(len(files), number)
	docID = 0
	for file in files:
		with open(file) as f:
			for line in f:
				words = line.split()
				wordID = number
				for word in words:
					for i,d in enumerate(dictionary):
						if d[0] = word:
							wordID = i
					if wordID<number:
						feature_matrix[docID,wordID] += words.count(word)
		docID = docID + 1
	return feature_matrix

positive_feature_matrix = feature_extraction()
negative_feature_matrix = feature_extraction()

class nBayesClassifier:
	def __init__(self, positive_feature_matrix, negative_feature_matrix):
		self.positive_feature_matrix = positive_feature_matrix
		self.negative_feature_matrix = negative_feature_matrix

	def train(self, positive_train_matrix, negative_train_matrix):
		positive_numTrainDocs = len(positive_train_matrix)
		negative_numTrainDocs = len(negative_train_matrix)
		positive_docvectors = ones(len(positive_train_matrix[0]))
		positive_nom = 2
		negative_docvectors = ones(len(negative_train_matrix[0]))
		negative_nom = 2
		for i in range(positive_numTrainDocs):
			positive_docvectors += positive_train_matrix[i]
			positive_nom += sum(positive_train_matrix[i])
		for i in range(negative_numTrainDocs):
			negative_docvectors += negative_train_matrix[i]
			negative_nom += sum(negative_train_matrix[i])
		self.positive_docvectors = log(positive_docvectors/positive_nom)
		self.negative_docvectors = log(negative_docvectors/negative_nom)
		self.P_positive = positive_numTrainDocs/(positive_numTrainDocs+negative_numTrainDocs)
		self.P_negative = negative_numTrainDocs/(positive_numTrainDocs+negative_numTrainDocs) 

	def test(self, test_data):
		predict = []
		for i in len(test_data):
			p1 = sum(self.positive_docvectors*test_data[i]) +log(self.P_positive)
			p2 = sum(self.negative_docvectors*test_data[i]) +log(self.P_negative)
			if p1>p2:
				predict.append(1)
			else:
				predict.append(-1)
		self.predict = predict

	def evaluation(self, test_label):
		n1 = 0
		n2 = 0
		n3 = 0
		for i in len(test_label):
			if test_label[i] == -1 and self.predict[i] = -1:
				n1 += 1
			if test_label[i] == 1 and self.predict[i] = -1:
				n2 +=1
			if test_label[i] == -1 and self.predict[i] = 1:
				n3 += 1
		SP = n1/(n1+n2)
		SR = n1/(n1+n3)
		self.eval = SP*SR*2/(SP+SR)

NBC = nBayesClassifier(positive_feature_matrix, negative_feature_matrix)
NBC.train(positive_feature_matrix, negative_feature_matrix)
test_data = feature_matrix[[:].[:]]
test_label = []
test_label.append([1]*  ,[-1]* )
NBC.test(test_data)
NBC.evaluation(test_data)
print(NBC.eval)

def linearClassfier(traindata, trainlabel, testdata, testlabel,lamda):
	X = np.mat(traindata)
	y = np.mat(trainlabel).T
	W = (X.T*X+lamda*np.eye(len(X[0]),len(X[0]))).I*X.T*y
	pred_labels = np.sign(np.mat(test_datas) * W)
    return pred_labels.T.tolist()[0]

def evaluations(test_label, predict_labels):
	n1 = 0
	n2 = 0
	n3 = 0
	for i in len(test_label):
		if test_label[i] == -1 and predict_labels[i] = -1:
			n1 += 1
		if test_label[i] == 1 and predict_labels[i] = -1:
			n2 +=1
		if test_label[i] == -1 and predict_labels[i] = 1:
			n3 += 1
	SP = n1/(n1+n2)
	SR = n1/(n1+n3)
	result = SP*SR*2/(SP+SR)
	return result
















