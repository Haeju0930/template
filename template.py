#PLEASE WRITE THE GITHUB URL BELOW!
#

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC


def load_dataset(dataset_path):
	data_frame = pd.read_csv(dataset_path)
	return data_frame

def dataset_stat(dataset_df):
  num1= dataset_df.shape[1]
  num2=dataset_df.target[0]
  num3=dataset_df.target[1]
  return num1,num2,num3

def split_dataset(dataset_df, testset_size):
  X_train, X_test, y_train, y_test = train_test_split(dataset_df, dataset_df.target, testset_size)
  return X_train, X_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
  data_frame = DecisionTreeClassifier()
  data_frame.fit(x_train,y_train)
  a = accuracy_score(y_test,data_frame.predict(x_test))
  b = precision_score(y_test,data_frame.predict(x_test))
  c = recall_score(y_test,data_frame.predict(x_test))
  return a,b,c

def random_forest_train_test(x_train, x_test, y_train, y_test):
  data_frame = RandomForestClassifier()
  data_frame.fit(x_train,y_train)
  a = accuracy_score(data_frame.predict(y_test),x_test)
  b = precision_score(data_frame.predict(y_test),x_test)
  c = recall_score(data_frame.predict(y_test),x_test)
  return a,b,c

def svm_train_test(x_train, x_test, y_train, y_test):
  data_frame = SVC()
  data_frame.fit(x_train,y_train)
  a = accuracy_score(y_test,data_frame.predict(x_test))
  b = precision_score(y_test,data_frame.predict(x_test))
  c = recall_score(y_test,data_frame.predict(x_test))
  return a,b,c

def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
