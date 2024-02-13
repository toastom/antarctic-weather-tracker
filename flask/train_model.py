#!/usr/bin/env python3

# Must activate the conda env in order to use pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import datetime

from query import Query

class WeatherModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
		# num in_features, num out_features
		self.linear = nn.Linear(50, 1)
	def forward(self, x):
		x, _ = self.lstm(x)
		# output from only the last timestep
		#x = x[:, -1]
		x = self.linear(x)
		return x

	
def impute(data):
	pass
	
# Normalize data by rescaling (min-max) to 0 -> 1 scale
def normalize(data, dtype="tmin"):
	normed_data = []
	
	# Temporarily fill in null values with something obviously wrong that won't affect min or max
	# This is just a quick fix because for normalizing you can't have null values
	# Replace with an impute() function later
	for j in data:
		if None in data:
			j = data.index(None)
			data[j] = 0.0000012345
	
	for i in data:
		n = (i - min(data)) / (max(data) - min(data))
		normed_data.append(n)
		
	return normed_data

def percent_missing(data):
	count = 0
	for i in tmin:
		if i is None:
			count = count + 1
	return (count / len(data)) * 100
	

# Create training and test sets
def create_sets(data, window_size):
	# For now just fill null values in data with some obvious outlier
	"""
	for i in data:
		if None in data:
			i = data.index(None)
			data[i] = -100.0
	"""
	
	train_size = int(0.7 * window_size)
	test_size  = window_size - train_size
	
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
		# Range of the feature to train on
		feature = dataset[i : i+lookback]
		
		# Range of the feature to target and predict
		# Still iterate through the whole 'feature' range for good measure,
		# but plus one to predict a new point
		target = dataset[i+1 : i+lookback+1]
		
		x.append(feature)
		y.append(target)
	return torch.tensor(x), torch.tensor(y)


# Get data from the database, for now just one datatype like min temp
# Split full dataset into training set and testing set (70% train, 30% test)

# Right now, past time window = 5 days
# 3 days for training, 2 days for testing
# Looking to predict only the very next day


q = Query()
q.send_query("SELECT dt, tmin FROM Days")
full_dataset = q.get_result_cols()
dates = full_dataset[0]
tmin  = full_dataset[1]

p_empty = percent_missing(tmin)
print("Percent of missing data in tmin: %.2f %% " % p_empty)

norm = normalize(tmin)

window_size = 5
tr, ts = create_sets(norm, window_size)
print(f"Window size: {window_size}")
print(f"Training set: {tr}")
print(f"Test set: {ts}")

# x = feature; y = target
lookback = 1
x_train, y_train = create_tensors(tr, lookback)
x_test, y_test = create_tensors(ts, lookback)
print(x_train.shape, y_train.shape)
print(x_test.shape,  y_test.shape)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# We have the dataset, now create the model
model = WeatherModel()
optimizer = optim.Adam(model.parameters())
# Loss function for calculating error and to train to minimize error
loss_func = nn.MSELoss() # Mean squared error

# I'm not sure here why we want to shuffle, since it's a time-based series
# The example had a lookback = 4 and batch_size = 8
loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=5)


# Then train the dataset. RNNs repeatedly loop to retrain the model

num_epochs = 2000
for epoch in range(num_epochs):
	# Tell the LSTM we're in training mode
	model.train()
	
	# hopefully replace the x and y here with more meaningful names later once I know better
	# Train on one batch at a time
	for x_batch, y_batch in loader:
		y_pred = model(x_batch)
		# Input our bad predictions, then train the model to match our expected values
		loss = loss_func(y_pred, y_batch)
		# Reset the optimizer's biases and weights and things
		optimizer.zero_grad()
		
		# Back-propagate the error over the hidden layer?
		# "hidden layer" = the nodes with weights and biases that make the full transfer function?
		loss.backward()
		optimizer.step()
		
	# Only validate the model on every 100th iteration
	if(epoch % 100 != 0):
		continue
	#If so, validate the model and tell the LSTM we're in validation mode now
	model.eval()
	
	# Not sure what this part does yet
	# Get RMSE (root mean squared error) of the training and test parts
	# To see how far away we are from the desired outcome
	with torch.no_grad():
		y_pred = model(x_train)
		train_rmse = np.sqrt(loss_func(y_pred, y_train))
		y_pred = model(x_test)
		test_rmse = np.sqrt(loss_func(y_pred, y_test))
	print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
	


	
# Plotting

# Format data for plotting
dates_plot = dates[len(dates) - window_size :]
# Add next date for new predicted point
dates_plot.append(datetime.date(2023, 11, 11))
norm_plot  = norm[len(norm) - window_size :]
# Extend by 1 for new point
norm_plot.append(np.nan)

with torch.no_grad():
	
	y_pred = model(x_train)
	#print(f"y_pred train: {y_pred}")
	y_pred = y_pred[:, -1]
	print(f"y_pred train: {y_pred}")
	train_plot = np.ones_like(norm_plot) * np.nan
	
	#train_plot[: len(norm_plot) - len(y_pred)] = y_pred[-1]
	train_plot[-1] = y_pred[-1]
	
	print(f"train pred: {train_plot}")
	
	y_pred = model(x_test)
	#print(f"y_pred test: {y_pred}")
	y_pred = y_pred[:, -1]
	print(f"y_pred test: {y_pred}")
	#y_pred = y_pred[:]
	test_plot = np.ones_like(norm_plot) * np.nan
	
	#test_plot[: len(norm_plot) - len(y_pred)] = y_pred[-1]
	test_plot[-1] = y_pred[-1]
	print(f"test pred: {test_plot}")


# Plot data

fig, ax = plt.subplots()
plt.plot(dates_plot, norm_plot, c="b")
ax.scatter(dates_plot, norm_plot, color="blue")
ax.scatter(dates_plot, train_plot, color="red")
ax.scatter(dates_plot, test_plot,  color="green")
ax.set_ylim([0, 1]) # Set ylim to normalized range [0, 1]
ax.set_xlabel("Dates")
plt.show()



