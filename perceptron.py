#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:45:22 2022

@author: juan

Perceptron
"""
import random


def loadDataset(fn):
    f = open(fn)
    lines = f.readlines()
    f.close()
    X = []
    Y = []
    for i in range(1,len(lines)):
        data = lines[i].strip().split(',')
        x = []
        for k in range(0,len(data)-1):
            x.append(float(data[k]))
        X.append(x)
        Y.append(float(data[-1]))
    return X, Y

def shuffleExamplesAndLabels(X,Y):
    Z = list(zip(X, Y))
    random.shuffle(Z)
    X, Y = zip(*Z)
    return X, Y


def getMetrics(model, X, Y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(0,len(X)):
        x = X[i]
        y = Y[i]
        y_hat = model.predict(x)
        if y_hat*y > 0 and y == 1:
            tp += 1
        if y_hat*y > 0 and y == -1:
            tn += 1
        if y_hat*y <= 0 and y == 1:
            fn += 1
        if y_hat*y <= 0 and y == -1:
            fp += 1
    if (tp+fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = -1
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = -1
    accuracy = (tp + tn)/(tp+fp+fn+tn)
    if precision >= 0 and recall >= 0:
        f1 = 2*precision*recall/(precision + recall)
    else:
        f1 = -1
    return (precision,recall,accuracy,f1,tp,fp,tn,fn)

class Perceptron():
    def __init__(self, alpha, numIter, ws):
        self.alpha = alpha
        self.numIter = numIter
        self.ws = ws
            
                
    def fit(self,trainX,trainY,verbose=False):
        X = trainX
        Y = trainY
        self.ws = []
        for i in range(0,len(X)):
            X[i] = [1] + trainX[i]
        for i in range(0,len(X[0])):
            self.ws.append(0.0)
        prev_error = 1.0
        for j in range(0,self.numIter):
            error = 0.0
            X, Y = shuffleExamplesAndLabels(X, Y)
            for i in range(0,len(Y)):
                y_i = Y[i]
                x_i = X[i]
                y_hat = 0.0
                for k in range(0,len(x_i)):
                    x_ik = x_i[k]
                    w_k = self.ws[k]
                    y_hat += w_k*x_ik
                y_hat = y_i*y_hat
                if y_hat <= 0:
                    error += 1.0
                    for k in range(0,len(self.ws)):
                        self.ws[k] = self.ws[k]+self.alpha*y_i*x_i[k]
            if error != prev_error:
                prev_error = error
                if verbose:
                    print("Training Error: " + str(error/len(Y)))
        print("Final training error: " + str(error/len(Y)))
        
        
    def evalTest(self, testX, testY, verbose=False):
        if len(testY)>0:
            error = 0.0
            for i in range(0,len(testY)):
                y_i = testY[i]
                x_i = testX[i]
                y_hat = self.ws[0]
                for k in range(0,len(x_i)):
                    y_hat += self.ws[k+1]*x_i[k]
                if y_hat*y_i <= 0:
                    error += 1.0
            if verbose:
                print("Test Error: " + str(error/len(testY)))
        return error/len(testY)


    def predict(self,X):
        result = self.ws[0]
        for i in range(0,len(X)):
            result += self.ws[i+1]*X[i]
        if result >= 0:
            return 1
        else:
            return -1
    
    
    def __str__(self):
        return "Ws: " + str(self.ws)
    
 
    
def crossFold(model, k, trainX, trainY):
    print ("Doing " + str(k) + "-fold cross validation.")
    error = 0.0
    trainX, trainY = shuffleExamplesAndLabels(trainX, trainY)
    fold_size = len(trainY)//k
    for i in range(0,k):
        a = i*fold_size
        b = i*fold_size+fold_size
        foldTrainX = list(trainX[0:a]) + list(trainX[b:])
        foldTrainY = trainY[0:a] + trainY[b:]
        foldTestX = trainX[a:b]
        foldTestY = trainY[a:b]
        model.fit(foldTrainX,foldTrainY)
        fold_error = model.evalTest(foldTestX, foldTestY)
        print ("fold " + str(i+1) + " error: " + str(fold_error))
        error += fold_error
    print ("Average error: " + str(error/k) + "\n")
            
       
#trainX, trainY = loadDataset("/Users/juan/UVG/tigo/fake_data_easy.csv")
#trainX, trainY = loadDataset("/Users/juan/UVG/tigo/fake_data_random.csv")
trainX, trainY = loadDataset("/Users/juan/UVG/tigo/fake_data_better.csv")
#trainX, trainY = loadDataset("/Users/juan/UVG/tigo/fake_data_raw.csv")
trainX, trainY = shuffleExamplesAndLabels(trainX, trainY)
validationX = trainX[:10]
validationY = trainY[:10]
trainX = trainX[10:]
trainY = trainY[10:]


model = Perceptron(0.1,10000,[])

crossFold(model, 10, trainX, trainY)

print("Training with all the data")
model.fit(list(trainX),list(trainY),verbose=False)
print("Model's " + str(model) + "\n")
print("Evaluating validation:")
model.evalTest(validationX, validationY,verbose=True)

print("Evaluating test:")
testX, testY = loadDataset("/Users/juan/UVG/tigo/fake_data_test.csv")
model.evalTest(testX, testY,verbose=True)

print("\nTraining Metrics")
precision,recall,accuracy,f1,tp,fp,tn,fn = getMetrics(model, trainX, trainY)
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))
print("Accuracy: " + str(accuracy))
print("Confusion Matrix: ")
print(tp,fp)
print(fn,tn)

print("\nTesting Metrics")
precision,recall,accuracy,f1,tp,fp,tn,fn = getMetrics(model, testX, testY)
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))
print("Accuracy: " + str(accuracy))
print("Confusion Matrix: ")
print(tp,fp)
print(fn,tn)

