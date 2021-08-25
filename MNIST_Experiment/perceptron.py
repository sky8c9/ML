import numpy as np
import pandas as pd
import random

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train_data = df_train.values
test_data = df_test.values

def linear_kernel(u, v):
    return np.dot(u,v) + 1

def polynomial_kernel(u,v,d):
    return (np.dot(u,v) + 1)**d

def exponential_kernel(u, v):
    sigma = 120
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

def fit_perceptron(inputData, kernel):
    y = inputData[:,0]
    X = inputData[:,1:].astype(np.float)
    N, D = X.shape
    w = np.zeros(D).astype(np.float)
    m = np.repeat(False, N)
    partial = int(N*0.98)
    for t in range(N):
        x_t = X[t]
        y_t = y[t]
        y_mistake = y[m]
        X_mistake = X[m]
        y_hat = compute_y_hat(x_t, y_mistake, X_mistake, kernel)
        if y_hat != y_t:
            m[t] = True
            w = w + (y_t * x_t)
    return w

def buildClassifier(inputData):
    r, c = np.shape(inputData)
    classifier = {}
    for i in range(9):
        for j in range(i+1,10):
            m = np.repeat(False, r)
            temp = np.copy(inputData);
            for k in range(r):
                if (temp[k,0] == i):
                    m[k] = True
                    temp[k,0] = 1
                elif (temp[k,0] == j):
                    m[k] = True
                    temp[k,0] = -1
            #classifier[i * 10 + j] = fit_perceptron(temp[m], linear_kernel)
            classifier[i * 10 + j] = fit_perceptron(temp[m], exponential_kernel)

    return classifier

def testing(testData, classifier):
    r, c = np.shape(testData)
    for i in range(r):
        res = []
        for key in classifier:
            sign = compute_y_hat2(classifier[key], testData[i])
            if sign > 0:
                if key < 10:
                    res.append(0)
                else:
                    res.append(key/10)
            elif sign < 0:
                res.append(key%10)
        yHat = np.argmax(np.bincount(res))
        print(yHat)

def normalize(input):
    r, c = np.shape(input)
    for i in range(c):
        val = np.sqrt(np.sum(input[:,i]**2))
        if (val != 0):
            input[:,i] = input[:,i]*1.0/val
    return input

#normalize(train_data[:,1:].astype(np.float))
#normalize(test_data.astype(np.float))
testing(train_data[:,1:], buildClassifier(train_data))
#testing(test_data, buildClassifier(train_data))
