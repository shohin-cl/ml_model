import streamlit as st
import pandas as pd
import numpy as np


st.title('Career prediction')

st.caption('Prediction was made based on 2.2k observations. Only people who are happy with their occupation are left for prediction. Also, not all variables are used, the model works only with chosen 109 variables. All used data is clean. Two models are built. First predicts only one class for each test case ( only one profession is recommended for each individual). The second one is a little bit more complicated, it predicts 3 top answers for each X (3 occupations are predicted for each user ). First model composes 63%. accuracy, while the accuracy of the second model is 80%.')

code = '''
# Basic libraries #
import math
import pandas as pd
import numpy as np

# Vizualization Libraries #
import matplotlib.pyplot as plt
import seaborn as sns

# Classification Algorithms #
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Regression Algorithms #
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# Others #
import time
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Reading datasets #

data = pd.read_excel('/content/drive/MyDrive/Unicef/careerdata_eng_merged q and st_scaled21.06.22.xlsx')
need = pd.read_excel('/content/drive/MyDrive/Unicef/final_variables.xlsx')


# Converting all unknowns to -1 #

def nan(n) :
    return n != n

for i in data.columns :
    for j in range (len(data)) :
        item = data[i][j]
        if nan(item) or item == '-1' or item == '-1.0' :
            data[i][j] = -1


# Leaving only variables that we chose #

ar = list(need['VarName'])
erase = []

for i in data.columns :
    if (i in ar) or i == 'nom_q_6421' or i == 'ord_T_q_6604_1' or i == 'ord_T_st_6604_1' :
        continue
    erase.append(i)
data.drop(erase,axis=1,inplace=True)

# Erasing all -1 occurences in our label #

erase = []

for i in range (len(data)):
    item = data['nom_q_6421'][i]
    if item == -1 :
        erase.append(i)
data.drop(erase,inplace=True)

data.index = range(0,len(data))



# Deleting users which are not happy with their jobs #

erase = []
for i in range (len(data)) :

    item1 = data['ord_T_q_6604_1'][i]
    item2 = data['ord_T_st_6604_1'][i]

    if item1 == -1 :
        item1 = item2
    
    if item1 != 1 and item1 != 2 :
        erase.append(i)

data.drop(erase,inplace=True)


# Performing Get Dummies for nominal data #

for i in data.columns :
    if ('nom' not in i) or i == 'nom_q_6421' :
        continue
    ###################
    dummy = pd.get_dummies( data[i], prefix = i )
    data = pd.concat( [data,dummy], axis=1 )
    data.drop( i, axis = 1, inplace = True )



# Dividing data into train and test #
target = 'nom_q_6421'

X = data.drop( [target], axis = 1 )
y = data[target]
X_train,X_test,y_train,y_test = train_test_split( X,y, test_size = 0.2, random_state = 0 )

X_train.index = range(len(X_train))
y_train.index = range(len(y_train))
X_test.index = range(len(X_test))
y_test.index = range(len(y_test))


tim = time.time()
xgb.fit(X_train,y_train)
t = time.time()-tim
predict = xgb.predict(X_test)

if accuracy_score(y_test,predict) > mx :
    optimal = i
    mx = accuracy_score(y_test,predict)



# Model which checks top 3 answers #
length = len(X_test)

correct_answers = 0

X_cur_train = X_train.copy()
y_cur_train = y_train.copy()

for i in range ( length ):

    item = X_test[i:i+1]
    counter = 3

    while ( counter ) :

        counter -= 1
        ###################################

        xgb = XGBClassifier(n_estimators = 50, use_label_encoder = False)
        xgb.fit( X_cur_train, y_cur_train )

        ####################################

        prediction = xgb.predict( item )

        if prediction == y_test[i] :
            correct_answers += 1
            break

        ####################################
        erase = []
        for j in range ( len(y_cur_train) ) :
            if y_cur_train[j] == prediction :
                erase.append(j)

        X_cur_train.drop( erase, inplace = True )
        y_cur_train.drop( erase, inplace = True )

        X_cur_train.index = range( len(X_cur_train) )
        y_cur_train.index = range( len(y_cur_train) )

        #####################################
    X_cur_train = X_train.copy()
    y_cur_train = y_train.copy()

    print("{} row's current accuracy -> {}".format(i,correct_answers/(i+1)) )'''
st.code(code, language='python')



accuracy_model = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 92.85714285714286,  93.33333333333333,  93.75, 94.11764705882353, 94.44444444444444,  94.73684210526315, 95, 90.47619047619048, 86.36363636363636, 86.95652173913043, 87.5, 84, 84.61538461538461, 85.18518518518519, 85.71428571428571, 86.20689655172413,  86.66666666666667, 87.09677419354839, 87.5, 87.87878787878788, 88.23529411764706, 88.57142857142857, 86.11111111111112, 86.48648648648649, 86.8421052631579, 84.61538461538461, 85, 85.36585365853658, 85.71428571428571,  83.72093023255814, 84.09090909090909, 82.22222222222222, 82.60869565217391, 82.97872340425532,  81.25, 81.63265306122449,  82, 80.3921568627451, 80.76923076923077, 79.24528301886793, 79.62962962962963, 78.18181818181819, 78.57142857142857, 78.94736842105263, 77.58620689655172, 77.96610169491526, 76.66666666666667,  77.04918032786885, 77.41935483870968, 77.77777777777778,  78.125, 78.46153846153846, 78.78787878787878, 77.61194029850746, 77.94117647058824, 78.2608695652174, 78.57142857142857,  78.87323943661971,  79.16666666666666, 78.08219178082192,  78.37837837837838, 78.66666666666666, 78.94736842105263,  77.92207792207793, 78.2051282051282, 77.21518987341772, 77.5,  76.54320987654321, 76.82926829268293, 77.10843373493976, 76.19047619047619, 75.29411764705882, 75.58139534883721, 75.86206896551724, 76.13636363636364, 76.40449438202247, 76.66666666666667, 76.92307692307693, 77.173913043478260, 77.41935483870968, 77.6595744680851, 77.89473684210526, 78.125, 78.35051546391752,  78.57142857142857, 78.78787878787878, 79, 78.21782178217822, 78.43137254901961, 77.66990291262136,  77.88461538461539, 78.0952380952381, 78.30188679245284, 78.50467289719626, 77.77777777777778,  77.98165137614679, 78.18181818181819, 78.37837837837838,  78.57142857142857, 78.76106194690266, 78.94736842105263, 78.2608695652174, 78.44827586206896, 78.63247863247863, 78.8135593220339, 78.99159663865546, 78.33333333333333, 
78.51239669421488, 78.68852459016393, 78.86178861788617, 79.03225806451613, 79.2, 79.36507936507936, 79.52755905511811,  78.90625, 79.06976744186046,  79.23076923076923,  79.38931297709924, 78.78787878787878, 78.94736842105263, 78.35820895522388, 77.77777777777778,  77.94117647058824, 78.1021897810219, 78.2608695652174, 78.41726618705036, 78.57142857142857
] 


st.line_chart(accuracy_model)

st.subheader('Final accuracy is 80%')