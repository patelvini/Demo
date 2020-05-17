import numpy as np
import pandas as pd
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import Custom_Model
from sklearn.model_selection import train_test_split

class PerformancePredictor:

	def __init__(self, data):
		self.data = data

	def func_1(self,x):
		if 0<x<40:
			return "Fail"
		else:
			return "Pass"

	def func_2(self,x):
		if x>90:
			return 'O'
		elif 80<x<=90:
			return 'A+'
		elif 70<x<=80:
			return 'A'
		elif 60<x<=70:
			return 'B+'
		elif 50<x<=60:
			return 'B'
		elif 40<=x<=50:
			return 'C'
		elif 33<=x<=39:
			return 'P'
		else:
			return 'F'


	def convert_data(self):
		self.data=self.data.replace(['group A','group B','group C','group D','group E'],[0,1,2,3,4])
		self.data=self.data.replace(["bachelor's degree", 'some college', "master's degree","associate's degree", 'high school', 'some high school'],
             [0,1,2,3,4,5])
		self.data=self.data.replace(['standard', 'free/reduced'],[0,1])
		self.data=self.data.replace(['none', 'completed'],[0,1])
		self.data=self.data.replace(['male','female'],[0,1])
		return self.data

	def predict_total_score(self, data):
		data['total_score'] = (data['math_score'] + data['reading_score'] + data['writing_score'])/3
		data['total_score'] = data['total_score'].astype(int)

		return data

	def predict_performance(self,data):
		data['performance'] = data['total_score'].apply(self.func_1)

		return data

	def predict_grade(self,data):
		data['grade'] = data['total_score'].apply(self.func_2)

		return data


if __name__ == '__main__':

	# reading dataset
	path = os.getcwd()
	path = os.path.dirname(path)

	data = pd.read_csv(path+'/Data/StudentsPerformance.csv')

	# renaming columns
	data = data.rename(columns={'parental level of education':'parental_level_of_education',
                      'test preparation course':'test_preparation_course',
                     'math score':'math_score','reading score':'reading_score','writing score':'writing_score'})

	m = PerformancePredictor(data)
	data = m.convert_data()
	data = m.predict_total_score(data)
	data = m.predict_performance(data)
	data = m.predict_grade(data)

	print(data)


	gender = data['gender'].values
	race = data['race/ethnicity'].values
	p_level = data['parental_level_of_education'].values
	lunch = data['lunch'].values
	test_prep = data['test_preparation_course'].values
	math = data['math_score'].values
	read = data['reading_score'].values
	write = data['writing_score'].values
	total_score = data['total_score'].values

	n = len(gender)
	x0 = np.ones(n)

	# X = np.array([x0, gender, race, p_level, lunch, test_prep, math, read, write]).T
	X = np.array([gender, race, p_level, lunch, test_prep, math, read, write]).T

	Y = np.array(total_score)

	alpha = 0.0001

	# initial coefficients
	B = np.array([0,0,0,0,0,0,0,0])

	x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=7)

	model = Custom_Model.LinearRegressor()

	# m,c = model.calculateSlopeAndIntercept(x_train, y_train)
	m, c = model.gradient_descent(x_train, y_train, B, alpha, 10000)

	print("\nModel Coefficients : ", m)
	print("\nY-intercept of the line : ", c[-1])

	predictions = model.predict(x_test, m).astype(int)

	print("\nPredictions : ",predictions)

	r2 = model.calculate_R_square(y_test, predictions)
	RMSE = model.RMSE(y_test, predictions)

	print("\nR-Squared value : ",r2)
	print("\nRoot Mean Squared Error : ",RMSE)

	plt.scatter(y_test,predictions,label='predictions',c='red',edgecolors="blue")

	plt.legend()
	plt.show()