from numpy import *
import os
import re


def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        data = bytearray(f.read()[14:])
        data = numpy.array(data)
        return data

def get_faces_data(dir):
    datas = []
    for imgFile in os.listdir(dir):
        datas.append(read_pgm(dir + "/" + imgFile))
    return datas

def get_all_face_data():
    all_face_data = []
    for i in range(1,41):
        all_face_data.append(get_faces_data("data/orl_faces/s" + str(i)))
    return all_face_data


def pca(dataMat, test_data, threshold):
	mean = mean(dataMat, axis = 0)
	remove_mean = dataMat-mean
	covariance = cov(remove_mean)
	eigVals, eigVetors = linalg.eig(mat(covariance))
	sortEigVals = sorted(eigVals)
	total = sum(eigVals)
	N = 1
	while sum(sorted(sortEigVals[-N:]))/total<threshold:
		N = N+1
	eigIndex = argsort(eigVals)
	eigIndex = eigIndex[-N:]
	reEigvectors - eigVetors[:,eigIndex]
	low_data_mat = remove_mean*reEigvectors
	low_test_faces = test_data*reEigvectors
	preictions = []
	for face in low_test_faces:
		distance = sum(square(face-low_data_mat), axis = 1)
		prediction = floor(argmin(distance)/8)+1
		predictions.append(prediction)

	return predictions

def evaluation(predictions):
	acc = 0
	for i in range(40):
		if predictions[i] = floor(i/2)+1:
			acc += 1
	return acc/80

face_datas = get_all_face_data()
data = numpy.array(face_datas)
train_data = data[:,0:8,:]
test_data = data[:,8:10,:]
train_data.reshape(8*40, 92*112)
test_data.reshape(2*40, 92*112)
predictions = pca(train_data, test_data, 0.99)
accuracy = evaluation(predictions)
print(accuracy)




