import streamlit as st
import numpy as np
import pandas as pd

st.title('Career prediction model 3')
st.caption('Training was made based on 1.3k observations. Only people who are happy with their occupation are left for training and prediction. Also, not all variables are used, the model works only with chosen 109 variables. All the NAs in data are synthetically filled with KNN imputing algorithm.  One model is built. It predicts only one class for each test case ( only one profession is recommended for each individual ). Next step to is to synthesize more data observations to boost the accuracy')


code = '''

import pandas as pd
dataset = pd.read_excel("dataset.xlsx", dtype = object)
# Necessary variables
need = pd.read_excel('VariablesTable.xlsx')
# Using the variable to separate the data
for i in dataset.columns:
  if '2201' in i:
    print(i)

# Dropping all observations that are related to school
erase = []
for i in range(len(dataset)) :

    if dataset['nom_q_2201'][i] in [4,5,6,7,8]:
      continue
    erase.append(i)

dataset.drop(erase,inplace=True)
# Leaving only variables that we chose #

ar = list(need['VarName'])
erase = []

for i in range (len(dataset)):
  try:
    item = dataset['nom_q_6421'][i]
    if item == -1 :
        erase.append(i)
  except:
    pass
dataset.drop(erase,inplace=True)

dataset.index = range(0,len(dataset))

!pip install catboost
from catboost import CatBoostClassifier, Pool
import pickle

erase = []
for i in dataset.columns :
    if (i in ar) or i == 'nom_q_6421' or i == 'ord_T_q_6604_1' or i == 'ord_T_st_6604_1' :
        continue
    erase.append(i)


dataset.drop(erase,axis=1,inplace=True)

# Converting all unknowns USing Sklearn
from sklearn.impute import KNNImputer
import numpy as np

imputer = KNNImputer(n_neighbors=3)
new_data = imputer.fit_transform(dataset)

#Filling gaps
counter=0
raw_n = 0
sue = []
for j in range(len(new_data[0])):
  raw_n +=1
  for i in range(len(new_data)):
    if not new_data[i][j]==int(new_data[i][j]):
      new_data[i][j]=np.round(new_data[i][j])

for i in range(len(new_data)):
  for j in range(27):
    print(new_data[i][j]," ",end= "")
  print("")



Y = []
for i in range(len(new_data)):
  Y.append(new_data[i][2])
  new_data[i][2]=0


# Backing up the dataset to the other variable
data = dataset.copy()
data.shape

from sklearn import preprocessing


columns = dataset.columns

for i in range(len(columns)):
  try:
    for j in range(len(dataset[columns[i]])):
      int(dataset[columns[i]][j])
  except:
    le = preprocessing.LabelEncoder()
    le.fit(dataset[columns[i]])
    dataset[columns[i]] = le.transform(dataset[columns[i]])


model = CatBoostClassifier(learning_rate=0.03,
                           custom_metric=['Accuracy'])


from sklearn.model_selection import train_test_split
Y=np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(new_data, Y, test_size=0.08, random_state=42)

X_train.shape

from sdv.tabular import GaussianCopula
import sdv

generator = GaussianCopula()
generator.fit(X_train)
eval_dataset = Pool(X_test,
                    y_test)



model.fit(X_train,
          y_train,
          eval_set=eval_dataset,
          verbose=False)
        


preds_proba = model.predict_proba(eval_dataset)
preds = model.predict(eval_dataset)


def find_top_n(arr, n = 3):
  tops = []
  best = 0
  for i in range(n):
    tops.append(0)
  for i in range(len(arr)):
    if arr[i]>best:
      best = arr[i]
      for j in range(len(tops)-1):
        tops[j]=tops[j+1]
      tops[len(tops)-1]=i
  return tops


  Y_test = []
for i in y_test:
  Y_test.append(i)


total= 0
for i in range(len(y_test)):
  if preds[i]==Y_test[i]:
    total+=1
print("correct" ,total,  " total: ",len(Y_test), " acc: ",total/len(Y_test)*100)



total = 0
for i in range(len(y_test)):
  if Y_test[i] in find_top_n(preds_proba[i], n =3):
    total+=1
print("correct" ,total,  " total: ",len(Y_test), " acc: ",total/len(Y_test)*100)


file = open('cat-boost-model', 'wb')
pickle.dump(model,file)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=15, random_state=0)
clf.fit(X_train, y_train)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

total= 0
for i in range(len(y_test)):
  if preds[i]==Y_test[i]:
    total+=1
print("correct" ,total,  " total: ",len(Y_test), " acc: ",total/len(Y_test)*100)

from sklearn.neighbors import KNeighborsClassifier
for i in range(1,99):
  neigh = KNeighborsClassifier(n_neighbors=i)
  neigh.fit(X_train,y_train)
  preds = neigh.predict(X_test)
  
  total= 0
  for j in range(len(y_test)):
    if preds[j]==Y_test[j]:
      total+=1
  print(i,"  ",total/len(y_test))

'''



st.code(code, language='python')

st.subheader('Final accuracy is 50%')