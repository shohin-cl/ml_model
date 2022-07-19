import streamlit as st
import numpy as np
import pandas as pd

st.title('Career prediction model 2')
st.caption('Training was made based on 1.3k observations. Only people who are happy with their occupation are left for training and prediction. Also, not all variables are used, the model works only with chosen 109 variables.  Two models are built. First predicts only one class for each test case ( only one profession is recommended for each individual ). The second one is a little bit more complicated, it predicts 3 top answers for each X (3 occupations are predicted for each user ). In this case here are NAN Data, Next step to is to synthesize data to boost the accuracy.')

code = '''
dataset = pd.read_excel("dataset.xlsx")
need = pd.read_excel('VariablesTable.xlsx')

for i in dataset.columns:
  if '2201' in i:
    print(i)


erase = []
for i in range(len(dataset)) :

    if dataset['nom_q_2201'][i] in [4,5,6,7,8]:
      continue
    erase.append(i)

dataset.drop(erase,inplace=True)

# Converting all unknowns to -1 #

def nan(n) :
    return n != n

for i in dataset.columns :
    for j in range (len(dataset)) :
      try:
        item = dataset[i][j]
        if nan(item) or item == '-1' or item == '-1.0' :
            dataset[i][j] = -1
      except:
        pass

# Leaving only variables that we chose #

ar = list(need['VarName'])
erase = []


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

from catboost import CatBoostClassifier, Pool
import pickle

erase = []
for i in dataset.columns :
    if (i in ar) or i == 'nom_q_6421' or i == 'ord_T_q_6604_1' or i == 'ord_T_st_6604_1' :
        continue
    erase.append(i)

dataset.drop(erase,axis=1,inplace=True)


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


Y = dataset["nom_q_3222code"]
dataset.drop(["nom_q_3222code"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, Y, test_size=0.7, random_state=42)

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

total = 0
for i in range(len(y_test)):
  if Y_test[i] in find_top_n(preds_proba[i], n =3):
    total+=1
print("correct" ,total,  " total: ",len(Y_test), " acc: ",total/len(Y_test)*100)
'''


st.code(code, language='python')


st.subheader('Final accuracy is 56,96%')