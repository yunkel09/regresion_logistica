

import numpy as np
import pandas as pd
from numpy import log,dot,exp,shape
# from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split  




X,y = make_classification(n_features=4)

vinos = pd.read_csv("wine.csv")

# convertir a 1 y 0 la etiqueta categórica
def dumificar(lista):
  k = (pd.get_dummies(pd.DataFrame(lista, columns = ["quality"]).
       quality)[['Good']].
       rename(columns = {'Good': 'quality'}).
       quality.
       to_list())
  return k


# cargar dataset
def cargarDataset(df):
    f = open("wine.csv")
    lines = f.readlines()
    f.close()
    X = []
    Y = []
    for i in range(0, len(lines)):
        if i == 0:
            etiquetas = lines[i].strip().split(",")
        else:
            data = lines[i].strip().split(",")
            x = []
            for k in range(0, len(data) - 1):
                x.append(data[k])
            X.append(x)
            Y.append(data[-1])
            # Y = dumificar(Y)
    return(X, dumificar(Y), etiquetas)

train_x, train_y, lab = cargarDataset("wine.csv")



     
     
dumificar(s)   
     
j = k[['Good']].rename(columns = {'Good': 'quality'}).quality.to_list()

j[['quality']].to_list()

j.rename(columns = {'Good': 'q'})
j.rename(columns={'Good': 'col_one'})


trainX, trainY = loadDataset("fake_data_better.csv")


dummy = pd.get_dummies(vinos.quality, drop_first = True)
vinos_d = pd.concat((vinos, dummy), axis = 1)
vinos_d.drop(['quality'], axis = 1, inplace=True)
vinos_d.rename(columns = {"Good": "quality"}, inplace=True)

y = vinos_d.quality.to_numpy()
X = vinos_d.loc[:, vinos_d.columns != "quality"].to_numpy()

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3)

def standardize(tr):
    for i in range(shape(tr)[1]):
        tr[:,i] = (tr[:,i] - np.mean(tr[:,i]))/np.std(tr[:,i])
        
def metricas(y,y_hat):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*precision*recall/(precision+recall)
    return (precision, recall, f1_score)
   
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
   
   
   
class LogidticRegression:
    def sigmoid(self,z):
        sig = 1/(1+exp(-z))
        return sig
    def initialize(self,X):
        weights = np.zeros((shape(X)[1]+1,1))
        X = np.c_[np.ones((shape(X)[0],1)),X]
        return weights,X
    def fit(self,X,y,alpha=0.1,iter=1000):
        weights,X = self.initialize(X)
        def cost(theta):
            z = dot(X,theta)
            cost0 = y.T.dot(log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y)
            return cost
        cost_list = np.zeros(iter,)
        for i in range(iter):
            weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list
    def predict(self,X):
        z = dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis
       
       
       
standardize(X_tr)
standardize(X_te)
obj1 = LogidticRegression()
model = obj1.fit(X_tr,y_tr, alpha=0.01)
y_pred = obj1.predict(X_te)
y_train = obj1.predict(X_tr)

print("\nTraining Metrics")
precision,recall,accuracy,f1,tp,fp,tn,fn = getMetrics(model, trainX, trainY)
print("Precision: " + str(precision))
print("Recall: " + str(recall))
print("F1: " + str(f1))
print("Accuracy: " + str(accuracy))
print("Confusion Matrix: ")
print(tp,fp)
print(fn,tn)




precision_tr, recall_tr, f1_tr = metricas(y_tr, y_train)
print("Precision: " + str(precision_tr))
print("Recall: " + str(recall_tr))
print("F1: " + str(f1_tr))


f1_score_te = metricas(y_te, y_pred)
print(f1_score_tr)
print(f1_score_te)
