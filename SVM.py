from numpy import *
import random
class optStruct:
	def __init__(self, dataMatIn, classLabels, C, toler, sigma):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.toler = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m, 1)))   #初始化一个m的列向量α
		self.b = 0
		self.eCache = mat(zeros((self.m, 2)))	#误差(Ei)缓存
		self.K = mat(zeros((self.m, self.m)))   #初始化一个存储核函数值得m*m维的K
		self.sigma = sigma
		if sigma == 0:
			self.K = self.dataMatin * self.dataMatin.T
		else:
			for i in range(self.m):
				for j in range(self.m):
					diff = X[i,:]-X[j,:]
					a = sum(mutiply(diff,diff))
					self.K[i,j] = exp(-1/2*a/sigma^2)

def calcEk(os,k):
	fxk = float(mutiply(os.alphas, os.labelMat).T*os.K[:,k]+b)
	Ek = fxk-yk

def selectJrand(i, m):
	j = i
	while j==i:
		j = np.int(random.uniform(0,m))
	return j

def selectJ(i, os, Ei):
	maxK = -1; maxDeltaE = 0; Ej = 0
    os.eCache[i] = [1, Ei]        		
	validEcachelist = nonzero(os.eCache[:, 0].A)[0]
	if (len(validEcachelist)) > 1:
		for k in validEcachelist:
			if k == i: continue
			Ek = calcEk(os, k)
			deltaE = abs(Ei - Ek)
			if (deltaE > maxDeltaE):      
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej
	else:                                
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
	    return j, Ej

def updateEk(os, k):
	Ek = calcEk(os, k)
	eCache[k] = [1,Ek]

def clipAlpha(aj, H, L):
	if aj>H:
		aj = H
	if aj<L:
		aj = L
	return aj

def inner(i, os):
	Ei = calcEk(os, i):
	if ((os.labelMat[i] * Ei < -os.toler) and (os.alphas[i] < os.C)) or ((os.labelMat[i] * Ei > os.toler) and (os.alphas[i] > 0)):
		j, Ej = selectJ(i, os, Ei)      									
		alphaIold = oS.alphas[i].copy()
		alphaJold = oS.alphas[j].copy()  
		if os.labelMat[i]! = os.labelMat[j]:
			L = max(0, alphaJold - alphaIold)
			H = min(C, C+alphaJold - alphaIold)
		else:
			L = max(0, alphaJold+alphaIold -c) 
			H = min(C, alphaJold+alphaIold)
		if L== H:
			print("L=H")
			return 0
		eta = 2*K[i,j]-K[i,i]-K[j,j]
		if eta >= 0: 
			print("eta>=0")
			return 0
		os.alphas[j] -= os.labelMat[j]*(Ei-Ej)*eta
		os.alphas[j] = clipAlpha(os.alphas[j], L, H)
		oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
		updateEk(os, i)
		updateEk(os, j)

		b1 = os.b - Ei - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.K[i, i] - os.labelMat[j] * (os.labelMat[j] - alphaJold) * os.K[i, j]
		b2 = os.b - Ei - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.K[i, j] - os.labelMat[j] * (os.labelMat[j] - alphaJold) * os.K[j, j]
		if 0<os.alphas[i] and os.alphas[i]<C:
			os.b = b1
		elif 0<os.alphas[j] and os.alphas[j]<C:
			os.b = b2
		else:
			os.b = (b1+b2)	/2
		return 1
	else:
		return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, sigma = 0):
	oS = optStruct(mat(dataMatIn), mat(classLabels).T, C, toler, sigma)
	iter = 0
	entireSet = True; alphaPairsChanged = 0
	while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		if entireSet:															
			for  i in range(oS.m):
				alphaPairsChanged += innerL(i, oS)
			print "fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged)
		else:																	
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0] 
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i, oS)
				print "non-bound, iter: %d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged)
		iter += 1
		if entireSet: entireSet = False
		elif (alphaPairsChanged == 0): entireSet = True
		print "iteration number: %d" % iter
	return oS.b, oS.alphas

def SVM(traindata, trainlabels, testdata, testlabels, C, sigma = 0, toler):
	b, alphas = smoP(traindata, trainlabels, C, toler, maxIter = , sigma)
	predict_labels = []
	sum = 0
	if sigma = 0:
		W  = sum(multiply(mutiply(alpha, trainlabels), traindata), axis =1)
		B = b*np.ones(1,(shape(testdata)[0]))
		predict_labels = np.sign(W*testdata.T+B)
	else:
		for i in range(len(testdata)):
			for j in range(len(traindata)):
				sum = sum+alphas[j]*trainlabels[j]*exp(-1/2*sum(multiply(testdata[i,:]-traindata[j,:]))/sigma^2)
			predict_labels.append(np.sign(sum))
			sum = 0
	return predict_labels
	
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

#positive_data is actually a feature_matrix
def cross_evaluations(positive_data, negative_data):
	n = len(positive_data)
	m = len(negative_data)
	accuracy = []
	trainlabels = []
	testlabels = []
	for k in range(5):
		positive_data = positive_data.copy()
		negative_data = negative_data.copy()
		positive_test = positive_data[k*floor(n/5):((k+1)*floor(n/5))]
		negative_test = negative_data[k*floor(m/5):((k+1)*floor(m/5))]
		positive_train = del positive_data[k*floor(n/5):((k+1)*floor(n/5))]
		negative_train = del negative_data[k*floor(m/5):((k+1)*floor(m/5))]
		trainlabels.append([1]*len(positive_train))
		trainlabels.append([-1]*len(negative_train))
		testlabels.append([1]*len(positive_test))
		testlabels.append([-1]*len(negative_test))
		positive_test = np.mat(positive_test)
		positive_train = np.mat(positive_train)
		negative_test = np.mat(negative_test)
		negative_train = np.mat(negative_train)
		traindata = vstack(positive_train, negative_train)
		testdata = vstack(positive_test, negative_test)







