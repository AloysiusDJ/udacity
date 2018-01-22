import numpy as np
import pandas as pd
from IPython.display import display

import visuals as vs

in_file='./titanic_data.csv'
full_data = pd.read_csv(in_file)
outcomes = full_data.loc[:,'Survived']
data = full_data.drop('Survived', axis=1)

def accuracy_score(truth, pred):
    if len(truth)==len(pred):
        return "Predictions have an accuracy of {:.2f}%.".format((truth==pred).mean()*100)
    else:
        return "Number of predictions does not mach number of outcomes"

def predictions_3(data):
    predictions=[]
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