import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


data = pd.read_csv("D:/hackathons/hr_analytics/train_LZdllcl.csv")
print(len(data))
fn=['age','length_of_service','avg_training_score']
hh=['age','length_of_service','avg_training_score','department','region','education','gender','recruitment_channel','no_of_trainings','previous_year_rating','KPIs_met >80%','awards_won?']

#scaler = StandardScaler()
#data=data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

for i in hh:
    if i not in fn:
        data[i]=data[i].fillna(data[i].mode()[0])
    else:
        data[i]=data[i].fillna(data[i].median())

op={}
l=['department','region','education','gender','recruitment_channel','no_of_trainings','previous_year_rating','KPIs_met >80%','awards_won?','is_promoted']
print(data)

for i in l:
    k=list(set(data[i]))
    k.sort()
    t=0
    for j in k:
        op[j]=t
        t+=1
    print(k)
    for j in range(len(data)):
        data[i].iloc[j]=k.index(data[i].iloc[j])

'''
p=list(set(data['Dependents']))
p.sort()
for j in k:
        op[j]=t
        t+=1
for j in range(len(data)):
    data['Dependents'].iloc[j]=p.index(data['Dependents'].iloc[j])

'''
print(data)


for i in l:
    data[i]=pd.to_numeric(data[i])
'''
for i in fn:
    data[i]=np.log(data[i])
'''
#data['sum_performance ']=data['previous_year_rating']+data['KPIs_met >80%']+data['awards_won?']
#data['']=data['']+data['']
#data['training_vs_score']=data['avg_training_score']/data['no_of_trainings']


x=data.drop(columns=['employee_id','age','gender','no_of_trainings','is_promoted'],axis=1)
#x.dropna(subset =["Gender","Married",'Credit_History','Dependents','Property_Area'], inplace=True)
'''
for i in fn:
    data[i]=scaler.fit(data[i])
'''

y=data['is_promoted']
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(data)
#print(x[:20])
#print(y[:10])

print(x.shape)
print(y.shape)


params = {}
params['learning_rate'] = 0.3
params['max_depth'] = 18
params['n_estimators'] = 10000
params['objective'] = 'binary'
params['boosting_type'] = 'gbdt'
params['subsample'] = 0.7
params['random_state'] = 42
params['colsample_bytree']=0.7
params['min_data_in_leaf'] = 55
params['reg_alpha'] = 1.7
params['reg_lambda'] = 1.11
params['class_weight']: {0: 0.2, 1: 0.8}

alg = lgb.LGBMClassifier(**params)

alg.fit(x, y, early_stopping_rounds=100, eval_set=[(x,y), (x_test, y_test)], eval_metric='binary_logloss', verbose=True)



l_t=['department','region','education','gender','recruitment_channel','no_of_trainings','previous_year_rating','KPIs_met >80%','awards_won?']
fr_t=pd.read_csv('D:/hackathons/hr_analytics/test_2umaH9m.csv')

for i in hh:
    if i not in fn:
        fr_t[i]=fr_t[i].fillna(fr_t[i].mode()[0])
    else:
        fr_t[i]=fr_t[i].fillna(fr_t[i].median())

for i in l_t:
    for j in range(len(fr_t)):
        try:
            fr_t[i].iloc[j]=op[fr_t[i].iloc[j]]
        except:
            fr_t[i].iloc[j]=0
print(fr_t)
for i in l_t:
    fr_t[i]=pd.to_numeric(fr_t[i])
'''
for i in fn:
    fr_t[i]=np.log(fr_t[i])
    '''
print(fr_t)
k=fr_t['employee_id']
'''
for i in fn:
    fr_t[i]=scaler.fit(fr_t[i])
    '''

#fr_t['sum_performance ']=fr_t['previous_year_rating']+fr_t['KPIs_met >80%']+fr_t['awards_won?']
#fr_t['training_vs_score']=fr_t['avg_training_score']/fr_t['no_of_trainings']
fr_t=fr_t.drop(columns=['age','gender','no_of_trainings','employee_id'],axis=1)
y_pred = alg.predict(fr_t)


rs={'employee_id':k,'is_promoted':y_pred}

rs_s=pd.DataFrame(rs)
print(rs_s[:20])

rs_s.to_csv('hr_3.csv', index=False)