import pandas as pd
import numpy as np

def distM(inputTr, inputT):
    dotP = -2 * np.dot(inputT, np.transpose(inputTr))
    sumSquaredTr = np.square(inputTr).sum(axis = 1)
    sumSquaredT = np.square(inputT).sum(axis = 1)
    return np.sqrt(dotP + sumSquaredTr + np.transpose(np.matrix(sumSquaredT)))

def main(inputTr, inputT, yTr, k):
    distanceM = distM(inputTr[:,1:], inputT[:1000,:])
    r,c = np.shape(inputT)
    yHat = np.zeros(r)
    for i in range(1000):
        sLabels = np.copy(yTr[np.argsort(distanceM[i,:])].flatten())
        yHat[i] = np.argmax(np.bincount(sLabels[0:k]))
        print (yHat[i])

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train_data = df_train.values
test_data = df_test.values
main(train_data, test_data, train_data[:,:1],4)