import numpy as np
import pandas as pd
import os
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import Modules.Custom_model
from sklearn.model_selection import train_test_split
import joblib
from PIL import Image
import time
import re
import Modules.SendMail
from openpyxl import Workbook
from openpyxl import load_workbook


class PerformancePredictor:

	def __init__(self, data):
		self.data = data

	def func_1(self,x):
		if 0<=x<33:
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

	def check_email(self, email):
		regex = '^\\w+([\\.-]?\\w+)*@\\w+([\\.-]?\\w+)*(\\.\\w{2,3})+$'
		if(re.search(regex,email)):
			return "Valid"
		else:
			return "Invalid Email!!"


	def convert_data(self, data):
		data = data.replace(["bachelor's degree", 'some college', "master's degree","associate's degree", 'high school', 'some high school'],
             [0,1,2,3,4,5])
		data = data.replace(['standard', 'free/reduced'],[0,1])
		data = data.replace(['none', 'completed'],[0,1])
		data = data.replace(['male','female'],[0,1])
		return data

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
	#path = os.path.dirname(path)

	data = pd.read_csv(path+'/Data/StudentsPerformance.csv')

	# renaming columns
	data = data.rename(columns={'parental level of education':'parental_level_of_education',
                      'test preparation course':'test_preparation_course',
                     'math score':'math_score','reading score':'reading_score','writing score':'writing_score'})

	p = PerformancePredictor(data)
	data = p.convert_data(data)
	data = p.predict_total_score(data)
	data = p.predict_performance(data)
	data = p.predict_grade(data)


	gender = data['gender'].values
	p_level = data['parental_level_of_education'].values
	lunch = data['lunch'].values
	test_prep = data['test_preparation_course'].values
	math = data['math_score'].values
	read = data['reading_score'].values
	write = data['writing_score'].values
	total_score = data['total_score'].values

	X = np.array([gender, p_level, lunch, test_prep, math, read, write]).T
	Y = np.array(total_score)
	alpha = 0.0001
	# initial coefficients
	B = np.array([0,0,0,0,0,0,0])

	x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=7)

	st.sidebar.title("About")
	st.sidebar.info("This is application to know what is the best way to improve student scores in the exam by knowing their performance before exam!!!!")

	st.title("Predict Student's Performance")

	image = Image.open(path+'/Images/students.jpg')
	st.image(image, caption="let's make students happy",use_column_width=True )
	

	st.sidebar.title("Train Model")
	if st.sidebar.button('Train'):
		st.sidebar.text("Training Model...")
		progress_bar = st.sidebar.progress(0)
		for percent_complete in range(100):
			time.sleep(0.1)
			progress_bar.progress(percent_complete+1)
		model = Modules.Custom_model.LinearRegressor()
		m, c = model.gradient_descent(x_train, y_train, B, alpha, 10000)

		st.sidebar.success("Model Trained Successfully!!")

	st.sidebar.title("Test Model")

	if st.sidebar.button('Test'):
		model = Modules.Custom_model.LinearRegressor()
		m, c = model.gradient_descent(x_train, y_train, B, alpha, 10000)
		predictions = model.predict(x_test, m).astype(int)
		st.sidebar.text("R2 value : ")
		st.sidebar.text(model.calculate_R_square(y_test, predictions))
		st.sidebar.text("RMSE : ")
		st.sidebar.text(model.RMSE(y_test, predictions))

		plt.scatter(y_test,predictions,label='predictions',c='red',edgecolors="blue")

		st.sidebar.pyplot()


		st.sidebar.success("We have very low value of RMSE score and a good R2 score. I guess our model is pretty good.")


	st.subheader("To check the performance please fill the below details correctly.")

	name_ip = st.text_input("Full name of student : ")

	roll_no_ip = st.number_input("Roll number : ",min_value=1)

	email_ip = st.text_input("Parent's email id : ")

	gender_ip = st.radio("Gender : ",("male","female"))
	
	p_level_ip = st.radio("Parental level of education : ",("bachelor's degree", "some college", "master's degree","associate's degree", "high school", "some high school"))
	
	lunch_ip = st.radio('Lunch : ',('standard', 'free/reduced'))
	
	test_prep_ip = st.radio('Test preparation course : ',('none', 'completed'))
	
	math_ip = st.number_input('Math score : ',min_value=0.0,max_value=100.0)	
	read_ip = st.number_input('Reading score : ',min_value=0.0,max_value=100.0)	
	write_ip = st.number_input('Writing score : ',min_value=0.0,max_value=100.0)

	

	if st.button("Predict Score"):
		msg = p.check_email(email_ip)

		if msg == "Valid":
			data_list = [[roll_no_ip, name_ip, email_ip, gender_ip, p_level_ip, lunch_ip, test_prep_ip, math_ip, read_ip, write_ip]]

			data = pd.DataFrame(data_list, columns = ['roll_no','name','email','gender', 'parental_level_of_education', 'lunch', 'test_preparation_course', 'math_score', 'reading_score', 'writing_score'])
		
			data = p.convert_data(data)

			gender = data['gender'].values
			p_level = data['parental_level_of_education'].values
			lunch = data['lunch'].values
			test_prep = data['test_preparation_course'].values
			math = data['math_score'].values
			read = data['reading_score'].values
			write = data['writing_score'].values

			X = np.array([gender, p_level, lunch, test_prep, math, read, write]).T

			model = Modules.Custom_model.LinearRegressor()
			m, c = model.gradient_descent(x_train, y_train, B, alpha, 10000)

			prediction = model.predict(X, m).astype(int)[0]

			progress_bar = st.progress(0)
			for percent_complete in range(100):
				time.sleep(0.1)
				progress_bar.progress(percent_complete+1)

			st.success("Done!!")

			data['total_score'] = prediction
			data['total_score'] = data['total_score'].astype(int)

			data = p.predict_performance(data)
			data = p.predict_grade(data)

			total_score = data['total_score'][0]
			Result = data['performance'][0]
			Grade = data['grade'][0]

			st.write("Total score : ",total_score,"%")

			st.write("Result : ",Result)
			
			st.write("Grade : ",Grade)

			if data['total_score'][0] >=85:
				st.success("**Congratulations! Keep going!!! Best of luck!!**")
			elif data['total_score'][0] >=50 and data['total_score'][0] < 85:
				st.success("**Nice! You can still do better. Best of luck!!**")
			else:
				st.warning("**Woops!!! You need to work hard. Don't loose hope and keep working hard. Best of luck!!**")


			sm = Modules.SendMail.Mail(name_ip, roll_no_ip, total_score, Result, Grade, email_ip)

			sm.send_mail()

			with st.spinner('!!Mail sent to given email address!!'):
				time.sleep(2)

			list_1 = [[roll_no_ip, name_ip, email_ip, gender_ip, p_level_ip, lunch_ip, test_prep_ip, math_ip, read_ip, write_ip, total_score, Grade, Result]]

			data_1 = pd.DataFrame(list_1, columns = ['Roll No.','Name','Parent\'s Email ID','Gender', 'Parental Level of  Education', 'Lunch', 'Test Preparation Course', 'Maths Score', 'Reading Score', 'Writing Score', 'Total Score', 'Grade', 'Result'])

			if os.path.exists(path+'/Data/Student_data.xlsx'):
				
				wb = load_workbook(path+'/Data/Student_data.xlsx')
				sheet = wb.active
				for row in list_1:
					sheet.append(row)
				wb.save(path+'/Data/Student_data.xlsx')
				
			else:
				data_1.to_excel(path+'/Data/Student_data.xlsx', index=False, header=True)

			st.success("Data Saved Successfully!! to "+path+"/Data/Student_data.xlsx")


		else:
			st.warning(msg)