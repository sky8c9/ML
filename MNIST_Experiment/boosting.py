import numpy as np
import pandas as pd

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train_data = df_train.values
test_data = df_test.values
def findCoeffW(wr):
    return 0.5 * np.log((1-wr)/float(wr))

def findCWAlpha(wr, flag):
    if (flag):
        return np.exp(-wr)
    else:
        return np.exp(wr)

def sigmoid(w, x):
    score = np.dot(w, np.transpose(x))
    return 1 / (1 + np.exp(-score))

def polynomial_kernel(u,v,d):
    return (np.dot(u,v) + 1)**d

def exponential_kernel(u, v):
    sigma = 119
    return np.exp(-np.linalg.norm(u-v)/(2*sigma**2))

def compute_y_hat2(w, x):
    def sign(x): return 1 if x >= 0 else -1
    return sign(np.dot(w,x))

def compute_y_hat(x_t, y_mistake, X_mistake, kernel):
    def sign(x): return 1 if x >= 0 else -1
    n_mistake = len(y_mistake)
    if not n_mistake:
        return sign(0)
    sum = 0
    for i in range (n_mistake):
        sum+=y_mistake[i] * kernel(X_mistake[i], x_t)
    return sign(sum)

def boosting(inputData, cSize, classes):
    y = inputData[:, 0]
    X = inputData[:, 1:].astype(np.float)
    N, D = X.shape

    w = np.zeros(D).astype(np.float)
    newW = np.copy(w)
    res = np.zeros(D).astype(np.float)
    start = 0

    WeakL = np.empty((cSize,D,))
    #size = int(N/cSize)
    size = 100
    count = 0

    for i in range(cSize):
        for m in range(start,start+size,1):
            yHat = sigmoid(w, X[count])
            for j in range(D):
                partial = X[m, j] * (max(0,y[count]) - yHat)
                newW[j] = w[j] + 0.1 * partial
            w = np.copy(newW)
            count+=1
        WeakL[i] = w
        start+=size

    alpha = np.zeros(N)
    alpha.fill(1.0 / N)
    index = 0

    for t in range(4):
        Rer = 100000
        Rwrong = np.repeat(False, N)
        Rcorrect = np.repeat(False, N)

        for g in range(cSize):
            wrong = np.repeat(False, N)
            correct = np.repeat(False, N)
            er = 0.0
            for l in range(N):
                y_Hat = compute_y_hat2(WeakL[g],X[l])
                if y_Hat != y[l]:
                    wrong[l] = True
                    er+=alpha[l]
                else:
                    correct[l] = True

            if (er < Rer):
                Rer = er
                Rwrong = wrong
                Rcorrect = correct
                index = g

        wr = findCoeffW(Rer)
        alpha[Rwrong] = alpha[Rwrong] * findCWAlpha(wr, 0)
        alpha[Rcorrect] = alpha[Rcorrect] * findCWAlpha(wr, 1)
        alpha = alpha / float(alpha.sum())
        res+=wr*WeakL[index]

    return res


def buildClassifier(inputData):
    r, c = np.shape(inputData)
    classifier = {}
    for i in range(10):
        temp = np.copy(inputData);
        for k in range(r):
            if (temp[k,0] == i):
                temp[k,0] = 1
            else:
                temp[k,0] = -1
        classifier[i] = boosting(temp,50,i)
    return classifier


def testing(testData, classifier):
    r, c = np.shape(testData)
    val = np.zeros(len(classifier)).astype(np.float)
    for i in range(r):
        res = []
        for key in classifier:
            curVal = sigmoid(classifier[key], testData[i])
            val[key] = np.exp(curVal)
        yHat = np.argmax(val/np.sum(val))
        print(yHat)


def normalize(input):
    r, c = np.shape(input)
    for i in range(c):
        val = np.sqrt(np.sum(input[:,i]**2))
        if (val != 0):
            input[:,i] = input[:,i]*1.0/val
    return input

def normalize2(input):
    r, c = np.shape(input)
    for i in range(r):
        input[i] = input[i]/np.linalg.norm(input[i])
    return input

s_test = test_data[:,:]
s_train = train_data[:1,:]

#normalize(train_data[:,1:].astype(np.float))
#Ntest = normalize(s_test.astype(np.float))
#testing(Ntest, buildClassifier(train_data))

testing(test_data, buildClassifier(train_data))