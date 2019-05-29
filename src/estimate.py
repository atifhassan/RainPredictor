import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

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
	
	
	#replace categorical values with numerical values without using dummies
	Locations = df['Location'].unique()
	for i in range(0, len(Locations)):
		df = df.replace({Locations[i]:i})
		
	WindGustDirs = df['WindGustDir'].unique()
	for i in range(0, len(WindGustDirs)):
		df = df.replace({'WindGustDir':{WindGustDirs[i]:i}})
		
	WindDir9ams = df['WindDir9am'].unique()
	for i in range(0, len(WindDir9ams)):
		df = df.replace({'WindDir9am':{WindDir9ams[i]:i}})
	
	WindDir3pms = df['WindDir3pm'].unique()
	for i in range(0, len(WindDir3pms)):
		df = df.replace({'WindDir3pm':{WindDir3pms[i]:i}})
	

	# Obtain x and y
	y = df.RainTomorrow
	x = df.drop(['RainTomorrow', 'RISK_MM'], axis=1)

	#replace yes and no with booleans
	y = y.replace({'No':0, 'Yes':1})
	x = x.replace({'No':0, 'Yes':1})

	#scale values
	scaler = preprocessing.MinMaxScaler(feature_range = (1,2))
	xScaled = pd.DataFrame(scaler.fit_transform(x),columns = x.columns)

	#split data
	xTrain, xTest, yTrain, yTest = train_test_split(xScaled, y)

	return xTrain, xTest, yTrain, yTest
	
def estimate(x):
	x2 = x.fillna(0)

	data = x2.to_numpy()
	
	n = 15
	
	#use K Means Clustering to group data entries
	kmeans = KMeans(n_clusters = n, random_state=69).fit(data)
	clusters = kmeans.labels_
	
	
	clustersDF = pd.DataFrame(clusters, columns=['Cluster'])
	
	x2 = pd.concat([x2, clustersDF], axis = 1)
	
	#group data entries by their clusters and calculate mean values of each cluster
	averages = x2.groupby(['Cluster']).mean().to_numpy().tolist()
	
	#replace unknown values by the mean value of the cluster
	for i in range(0, len(data)):
		for j in range(0, len(data[i])):
			if data[i][j] == 0:
				cluster = clusters[i]
				data[i][j] = averages[cluster][j]
				
	dataDF = pd.DataFrame(data, columns=x.columns)
		
			
	
	
	return dataDF
	
	
def featureSelection(x, y):
	selector = SelectKBest(chi2).fit(x, y)
	scores = selector.scores_
	
	columns = x.columns
	
	featureSelect = [0]
	
	for i in range(0, len(scores)):
		featureSelect.append((columns[i], scores[i]))
	
	#removes filler 0 used at initialization of list
	featureSelect.pop(0)
	featureSelect.sort(key=lambda x:x[1], reverse=True)
	
	'''
	for i in range(0,len(featureSelect)):
		print(featureSelect[i])
	print('\n')
	'''
	
	important = ['RainToday','Cloud3pm', 'Sunshine','Cloud9am','Humidity3pm','Rainfall','Humidity9am','WindGustSpeed','Temp3pm','Pressure9am']
	
	x.drop(x.columns.difference(important), axis=1, inplace=True)
	
	return x
	
def featureSelectionTest(x):
	
	important = ['RainToday','Cloud3pm', 'Sunshine','Cloud9am','Humidity3pm','Rainfall','Humidity9am','WindGustSpeed','Temp3pm','Pressure9am']
	
	x.drop(x.columns.difference(important), axis=1, inplace=True)
	
	return x
	

	
def main():
	start = time.time()
	
	xTrain, xTest, yTrain, yTest = preprocess()
	
	#estimate unknown values
	xTrainEst = estimate(xTrain)
	xTestEst = estimate(xTest)
	
	#feature selection
	#xTrainFinal = featureSelection(xTrainEst, yTrain)
	#xTestFinal = featureSelectionTest(xTestEst)
	
	#rebalance dataset using SMOTE
	#sm = SMOTE(random_state=12, ratio = 1.0)
	#x_train_res, y_train_res = sm.fit_sample(xTrainEst, yTrain)
	
	dtc = DecisionTreeClassifier()
	dtc.fit(xTrainEst, yTrain)
	
	accuracy = dtc.score(xTestEst, yTest)
	
	end = time.time()
	print('Accuracy: ' + str(accuracy) + '\n')
	print('Runtime: ' + str(end - start) + ' seconds')
	
	return
	
main()
	
	