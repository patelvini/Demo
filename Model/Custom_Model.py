import numpy as np

class LinearRegressor:

	def cost_function(self, X, Y, B):

		m = len(Y)
		J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
		return J

	def gradient_descent(self, X, Y, B, alpha, iterations):

		cost_history = [0] * iterations
		m = len(Y)

		for iteration in range(iterations):
			# Hypothesis values
			h = X.dot(B)

			# Difference between Hypothesis and Actual Y
			loss = h - Y

			# Gradient Descent
			gradient = X.T.dot(loss) / m

			# Changing Values of B using Gradient
			B = B - alpha * gradient

			# New Cost Value
			cost = self.cost_function(X, Y, B)
			cost_history[iteration] = cost

		return B, cost_history


	def predict(self, X, m):

		Y_pred = X.dot(m)

		return Y_pred


	def calculate_R_square(self, Y_test, Y_pred):

		mean_y = np.mean(Y_test)

		ss_tot = sum((Y_test - mean_y)**2)
		ss_res = sum((Y_test - Y_pred)**2)

		r2 = 1 - (ss_res/ss_tot)

		return r2

	def RMSE(self, Y_test, Y_pred):

		rmse = np.sqrt(sum(Y_test - Y_pred)**2)/len(Y_test)
		return rmse

