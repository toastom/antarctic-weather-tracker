#!/usr/bin/env python3

# Must activate the conda env in order to use pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
import matplotlib.pyplot as plt
import numpy as np
import datetime
import sys

from query import Query

class WeatherModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.hidden_layers = 50
		self.output_size = 1
		# can change lookback as long as input_size is the same
		self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_layers, num_layers=1, batch_first=True)
		
		self.linear = nn.Linear(self.hidden_layers, self.output_size)

	def forward(self, x):
		lstm_out, _ = self.lstm(x.view(len(x) ,1, -1))
		preds = self.linear(lstm_out.view(len(x), -1))
		
		return preds
	

model = WeatherModel()

# NOTE: PLEASE fix this later I know this isn't a terribly efficient solution.
# if any employers are here looking at this please know I'm better than this lol
# ALSO this is a pretty trash imputation algo anyway, just finding the midpoint between the closest two non-null points
def impute(data_points):
	imputed_data = [x for x in data_points]
	
	# For debugging imputation
	#new_points = [None for x in data_points]
	
	if(None not in data_points):
		return data_points
	
	last_non_null = None
	for i in range(0, len(data_points)):
		next_non_null = None
		
		if(data_points[i] != None):
			last_non_null = data_points[i]
		else:
			# Get the next defined point on the right side of this point
			j = i
			
			while(next_non_null == None and j < len(data_points)):
				if(data_points[j] != None):
					next_non_null = data_points[j]
				j = j + 1
				# Catch an edge case if we're at the end of the list
				# and we haven't found a non-null yet, loop back around
				if(j == len(data_points)):
					j = 0
					continue
			
			# Make sure last_non_null gets defined by looping through the left side
			k = i
			while(last_non_null == None and k > -len(data_points)):
				if(data_points[k] != None):
					last_non_null = data_points[k]
				k = k - 1
				# Catch an edge case if we're at the end of the list
				# and we haven't found a non-null yet, loop back around
				if(k == -len(data_points)):
					k = 0
					continue
			
			imputed_data[i] = (last_non_null + next_non_null) / 2
		
	# For debugging imputation
	#print(f"Data points: {data_points[len(data_points)-101:]}")
	#print(f"Imputed data: {imputed_data[len(imputed_data)-101:]} \n")
	"""
	fig, ax = plt.subplots()
	plt.plot(range(len(data_points)-51, len(data_points)), data_points[len(data_points)-51:], c="b")
	plt.plot(range(len(new_points)-51, len(new_points)), new_points[len(new_points)-51:], c="r")
	#plt.plot(range(0, len(data_points)), data_points, c="b")
	#plt.plot(range(0, len(new_points)), new_points, c="r")
	
	ax.scatter(range(len(data_points)-51, len(data_points)), data_points[len(data_points)-51:], color="blue")
	ax.scatter(range(len(new_points)-51, len(new_points)), new_points[len(new_points)-51:], color="red")
	#ax.scatter(range(0, len(data_points)), data_points, color="blue")
	#ax.scatter(range(0, len(new_points)), new_points, color="red")
	plt.show()
	"""
		
	return imputed_data
	
# Normalize data by rescaling (min-max) to 0 -> 1 scale
def normalize(data_points):
	normed_data = []
	imputed_data = impute(data_points)
	
	for i in imputed_data:
		n = (i - min(imputed_data)) / (max(imputed_data) - min(imputed_data))
		normed_data.append(n)
		
	return normed_data

def denormalize(data_points, mini, maxi):
	denormed_data = []
	for j in data_points:
		#d = (j - mini) / (maxi - mini)
		d = (maxi - mini) * j + mini
		denormed_data.append(d)
		
	return denormed_data

# Create training and test sets
def create_sets(data, window_size):
	window_size = window_size + 1
	train_size = int(0.7 * window_size) + 1
	test_size  = window_size - train_size + 1
	
	# from the start of the window up to end of the training window
	train_set = data[len(data) - window_size : len(data) - window_size + train_size]
	# remaining spots in the data are for our test window
	test_set  = data[len(data) - test_size : ]
	
	return train_set, test_set


# Turn weather dataset into Pytorch tensors
def create_tensors(dataset, lookback):
	x = []
	y = []
	for i in range(0, len(dataset) - lookback):
		feature = dataset[i : i+lookback]
		target = dataset[i+1 : i+lookback+1]
		
		x.append(feature)
		y.append(target)
	return torch.tensor(x), torch.tensor(y)


def train_model(num_epochs, x_train, y_train, x_test, y_test):
	optimizer = optim.Adam(model.parameters())
	# MSE = Mean Squared Error
	loss_func = nn.MSELoss()
	# Batch size must = 1 right now or else we get a dimension error
	# Not sure if shuffle should = True or False
	loader = td.DataLoader(td.TensorDataset(x_train, y_train), shuffle=True, batch_size=1)

	y_pred = []
	# Train the model
	for epoch in range(num_epochs):
		# Tell the LSTM we're in training mode
		model.train()

		for data, target in loader:
			y_pred = model(data)
			# Input our predictions, then train the model to match our expected values
			loss = loss_func(y_pred, target)
			# Reset the optimizer's biases and weights
			optimizer.zero_grad()

			# Back-propagate the error over the hidden layer
			loss.backward()
			optimizer.step()

		if(epoch % 100 != 0):
			continue
		model.eval()

		# Get MSE (mean squared error) of the training and test parts
		# To see how far away we are from the desired outcome
		with torch.no_grad():
			y_pred = model(x_train)
			train_rmse = np.sqrt(loss_func(y_pred, y_train))

			y_pred = model(x_test)
			test_rmse = np.sqrt(loss_func(y_pred, y_test))
		print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
		
	return y_pred

	
def main(datatype, window_size, num_epochs, future_preds):
	# Input handling
	if(int(sys.argv[3]) <= 0):
		raise Exception("Input argument 4 'future_preds' must be > 0")
	
	q = Query()
	q.send_query("DESCRIBE Days")
	possible_types = q.get_result_cols()[0]
	if((datatype not in possible_types) or (datatype == "dt")):
		err = [e for e in possible_types if(e != "dt")]
		raise Exception("Input argument 1 'datatype' must only be one of:\n" + str(err))
	
	# Setup to train the model 
	q.send_query(f"SELECT dt, {datatype} FROM Days")
	query_results = q.get_result_cols()
	dates = query_results[0]
	dataset  = query_results[1]

	norm = normalize(dataset)

	tr, ts = create_sets(norm, window_size)
	print(f"Window size: {window_size}")

	# x = feature; y = target
	lookback = 1
	x_train, y_train = create_tensors(tr, lookback)
	x_test, y_test = create_tensors(ts, lookback)
	print("x_train shape: ", x_train.shape, "\ty_train shape: ", y_train.shape)
	print("x_test shape: ", x_test.shape, "\ty_test shape: ", y_test.shape)
	# Training
	y_pred = train_model(num_epochs, x_train, y_train, x_test, y_test)
	print("y_pred shape: ", y_pred.shape)
	print("y_pred: ", y_pred)
	
	# After training + validation, make future predictions
	model.eval()
	predictions = torch.tensor([])

	with torch.no_grad():
		train_tens = torch.FloatTensor(tr)
		test_tens  = torch.FloatTensor(ts)

		model_train = model(train_tens)
		model_test  = model(test_tens)
		predictions = torch.cat((predictions, model_train), dim=0)
		predictions = torch.cat((predictions, model_test),  dim=0)

		# Formating data for future predictions
		# Model already predicts 2 future dates by default, so remove extra points if we want less than two future preds
		# otherwise, expand the tensor with future points
		#pred_tens = torch.FloatTensor(predictions[-future_preds+2:])
		if(future_preds > 2):
			pred_tens = torch.FloatTensor(predictions[-future_preds : -2])
			predictions = torch.cat((predictions, model(pred_tens)), dim=0)		
		elif(future_preds == 1):
			predictions = predictions[ : -1]
		
	# Format data for plotting
	dates_plot = dates[len(dates) - window_size :]
	day = datetime.timedelta(days=1)
	norm_plot  = norm[len(norm) - window_size :]

	# Add more days to the plot arrays for future predictions
	for i in range(1, future_preds+1):
		dates_plot.append(datetime.date(2023, 11, 10) + i*day)
		norm_plot.append(np.nan)
	# Undo the normalization to also plot the real units (deg F or inches)
	new_dataset = impute(dataset)
	real_values = denormalize(norm_plot, min(new_dataset), max(new_dataset))
	denorm_preds = denormalize(predictions.numpy(), min(new_dataset), max(new_dataset))

	print(f"shape predictions = {predictions.shape}")
	print(f"len dates = {len(dates_plot)}")

	torch.save(model.state_dict(), "weather_model.pt")

	# Plot data
	fig, ax = plt.subplots()
	plt.plot(dates_plot, real_values, c="g")
	plt.plot(dates_plot, denorm_preds, c="r")
	ax.scatter(dates_plot, real_values, color="green")
	ax.scatter(dates_plot, denorm_preds, color="red")
	ax.set_ylim([min(real_values), max(real_values)])
	ax.set_xlabel("Date")

	# Plot data
	fig, ax = plt.subplots()
	plt.plot(dates_plot, norm_plot, c="b")
	plt.plot(dates_plot, predictions, c="r")
	ax.scatter(dates_plot, norm_plot, color="blue")
	ax.scatter(dates_plot, predictions, color="red")
	ax.set_ylim([0, 1])
	ax.set_xlabel("Date")
	
	plt.show()


if(len(sys.argv) < 5):
	raise Exception("Must input arguments for:\ndatatype (str),\nwindow size (int),\nnum training epochs (int),\nnum future predictions (int)")

# EXAMPLE: tmin, 50, 500, 5
# datatype, window size, # of training epochs, # of future date predictions to generate
# datatype one of tmin, tmax, tavg, prcp, snwd
# NOTE: prcp and snwd preds will probably not be that accurate due to the amount of missing data
main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))


