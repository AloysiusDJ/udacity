# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 06:41:20 2018

@author: Aloysius Joseph
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
from IPython.display import display

import visuals as vs

log_dir="./graph"
shutil.rmtree(log_dir, ignore_errors=True)

#%matplotlib inline

in_file='./titanic_data.csv'
full_data = pd.read_csv(in_file)

#type(full_data)
#dir(full_data)
#display(full_data.info())
#display(full_data.head())

outcomes = full_data['Survived']
#outcomes = full_data.loc[:,'Survived']

data = full_data.drop('Survived', axis=1) #column axis=1

def accuracy_score(truth, pred):
    if len(truth)==len(pred):
        return "Predictions have an accuracy of {:.2f}%.".format((truth==pred).mean()*100)
    else:
        return "Number of predictions does not mach number of outcomes"


#predictions = pd.Series(np.ones(5, dtype=int))
#print(accuracy_score(outcomes[:5], predictions))

def predictions_0(data):
    predictions=[]
    for _,passenger in data.iterrows():
        predictions.append(0)
    return pd.Series(predictions)

predictions=predictions_0(data)
print(accuracy_score(outcomes, predictions))

vs.survival_stats(data, outcomes, 'Sex')

def predictions_1(data):
    predictions=[]
    for _,passenger in data.iterrows():
        if passenger['Sex']=='female':
            predictions.append(1)
        else:
            predictions.append(0)
            
    return pd.Series(predictions)

predictions=predictions_1(data)
print(accuracy_score(outcomes, predictions))

vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])
vs.survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Age <= 10"])

def predictions_2(data):
    predictions=[]
    for _,passenger in data.iterrows():
        if passenger['Sex']=='female' or (passenger['Sex']=='male' and passenger['Age']<=10):
            predictions.append(1)
        else:
            predictions.append(0)
            
    return pd.Series(predictions)
predictions=predictions_2(data)
print(accuracy_score(outcomes, predictions))

vs.survival_stats(data, outcomes, 'Age')
vs.survival_stats(data, outcomes, 'Sex')
vs.survival_stats(data, outcomes, 'Parch')
vs.survival_stats(data, outcomes, 'SibSp')

vs.survival_stats(data, outcomes, 'Sex', ["Sex == 'female'", "Age <= 10", "Parch == 0", "SibSp == 0"])

print(len(full_data))
print(len(full_data.query("Survived==1")))
print(full_data.query("Sex=='female' and Survived==0"))
print(len(full_data.query("Sex=='male' and Age>6 and Survived==1")))
print(len(full_data.query("Parch==0 and Survived==1")))
print(len(full_data.query("SibSp==0 and Survived==1")))
print(len(full_data.query("(Sex=='female' or (Sex=='male' and (not Age=='None' or Parch==0 or SibSp==0))) and Survived==1")))

print(len(full_data.query("Survived==1")))
pred = full_data.query("Sex=='female' | (Sex=='male' & Age<=10)");
print(len(pred))
tru = full_data.query("Sex=='female' | (Sex=='male' & Age<=10) & Survived==1");
print(len(tru))
print("{:.2f}%".format((tru==pred).mean()*100))

new_data2 = full_data.query("Sex=='female' | (Sex=='male' & Age<=10) & Parch==0 & SibSp==0");
print(len(new_data2))

print(full_data.head())
print(new_data.head())

in_file='./titanic_data.csv'
full_data = pd.read_csv(in_file)
full_data.info()
outcomes = full_data.loc[:,'Survived']
data = full_data.drop('Survived', axis=1) #column axis=1

def predictions_3(data):
    predictions=[]
    #print(type(data.query("Age <= 10")))
    for _,passenger in data.iterrows():
        if passenger['Age']<=6.0 or \
           (passenger['Sex']=='female' and (passenger['Pclass']==2 or passenger['Pclass']==1)) or \
           (passenger['Sex']=='female' and passenger['Parch']==0) or \
           (passenger['Sex']=='female' and passenger['SibSp']==0):
            predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)

predictions=predictions_3(data)
print(accuracy_score(outcomes, predictions))

"""
        if passenger['Age']<=6.0 or \
           (passenger['Sex']=='female' and (passenger['Pclass']==2 or passenger['Pclass']==1)) or \
           (passenger['Sex']=='female' and passenger['Parch']==0) or \
           (passenger['Sex']=='female' and passenger['SibSp']==0):
               """
               
predictions.name='Survived'
print(len(predictions))
print(len(outcomes))


print(sum(map(lambda x:x==0,outcomes)))
print(sum(map(lambda x:x==1,outcomes)))
print(sum(map(lambda x:x==0,predictions)))
print(sum(map(lambda x:x==1,predictions)))
print(sum(map(lambda x:x==False,outcomes==predictions)))
print(sum(map(lambda x:x==True,outcomes==predictions)))

print((pd.concat([outcomes, predictions, (outcomes==predictions)], axis=1)))
print("{:.2f}%".format((outcomes==predictions).mean()*100))

print(accuracy_score(outcomes, predictions))

print(outcomes.head(20))
print(predictions.head(20))
print((outcomes==predictions).head(200))

print("{:.2f}%".format((outcomes==predictions).mean()*100))
    

#print(full_data.head())

print(accuracy_score(outcomes, predictions))

full_data.info()

with tf.Session() as sess:
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    print(sess.run(full_data))
    writer.close()





