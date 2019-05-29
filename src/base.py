import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

import time

def preprocess():
	df = pd.read_csv('weatherAUS.csv');
	
	#convert 'Date' feature values into viable format
	dates = df['Date'].tolist()
	
	for i in range(0, len(dates)):
		dates[i] = dates[i].split('-')
		dates[i] = dates[i][1]
		
	newDates = pd.DataFrame({'Date':dates})
	df.update(newDates)

	# Obtain x and y
	y = df.RainTomorrow
	x = df.drop(['RainTomorrow', 'RISK_MM'], axis=1)

	#replace yes and no with booleans
	y = y.replace({'No':0, 'Yes':1})
	x = x.replace({'No':0, 'Yes':1})
	x = x.fillna(0)
	
	#transform categorical data into numerical data
	dummies = pd.get_dummies(data=x, columns=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
	
	#replace categorical data with their numerical representations
	x = x.drop(['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1)
	x = pd.concat([x, dummies], axis=1)
	

	#scale values
	scaler = preprocessing.MinMaxScaler(feature_range = (1,2))
	xScaled = pd.DataFrame(scaler.fit_transform(x),columns = x.columns)

	#split data
	xTrain, xTest, yTrain, yTest = train_test_split(xScaled, y)

	return xTrain, xTest, yTrain, yTest
	
def main():
	start = time.time()
	
	xTrain, xTest, yTrain, yTest = preprocess()
	
	print(len(xTrain))
	print(len(xTest))
	
	#rebalance dataset using SMOTE
	#sm = SMOTE(random_state=12, ratio = 1.0)
	#x_train_res, y_train_res = sm.fit_sample(xTrain, yTrain)
	
	dtc = DecisionTreeClassifier()
	
	dtc.fit(xTrain, yTrain)
	
	accuracy = dtc.score(xTest, yTest)
	
	end = time.time()
	print('Accuracy: ' + str(accuracy) + '\n')
	print('Runtime: ' + str(end - start) + ' seconds')
	
	return
	
main()
	
	