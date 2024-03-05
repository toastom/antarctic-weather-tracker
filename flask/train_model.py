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
		self.hidden_layers = 50
		self.output_size = 1
		self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_layers, num_layers=1, batch_first=True)
		
		# hidden layer size, output_size
		self.linear = nn.Linear(self.hidden_layers, self.output_size)

	def forward(self, x):
		#x, _ = self.lstm(x)
		#x = self.linear(x)
		#return x
		
		lstm_out, _ = self.lstm(x.view(len(x) ,1, -1))
		preds = self.linear(lstm_out.view(len(x), -1))
		
		#return preds[-1]
		return preds[0]

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
			# Last non-null point has already been acquired
			# Get the next defined point on the right side of this point
			j = i
			while(next_non_null == None and j < len(data_points)):
				if(data_points[j] != None):
					next_non_null = data_points[j]
				j = j + 1
				""" Idk if this will eventually be needed. Prob catches an edge case
				# if we're at the end and haven't found one yet, loop back around
				if(j == len(data_points)):
					j = 0
					continue
				"""
			
			imputed_data[i] = (last_non_null + next_non_null) / 2
			#new_points[i] = imputed_data[i]	
		
	#print(f"Data points: {data_points[len(data_points)-11:]}")
	#print(f"Imputed data: {imputed_data[len(imputed_data)-11:]} \n")
	
	# For debugging imputation
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
def normalize(data_points, dtype="tmin"):
	normed_data = []
	
	imputed_data = impute(data_points)
	
	for i in imputed_data:
		n = (i - min(imputed_data)) / (max(imputed_data) - min(imputed_data))
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


q = Query()
q.send_query("SELECT dt, tmin FROM Days")
full_dataset = q.get_result_cols()
dates = full_dataset[0]
tmin  = full_dataset[1]

#print(f"--- TMIN ---\n\n{tmin[len(tmin)-51:]}\n---\n")
"""
p_empty = percent_missing(tmin)
print("Percent of missing data in tmin: %.2f %% " % p_empty)
print("Length of tmin: " + str(len(tmin)))
"""
norm = normalize(tmin)

window_size = 50
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
loss_func = nn.MSELoss() # MSE = Mean Square Error

# Bug? Batch size must = 1 right now or else the model just converges its predictions
loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle=True, batch_size=1)


# Then train the dataset. RNNs repeatedly loop to retrain the model

num_epochs = 1000
for epoch in range(num_epochs):
	# Tell the LSTM we're in training mode
	model.train()
	
	for x_batch, y_batch in loader:
		y_pred = model(x_batch)
		# Input our bad predictions, then train the model to match our expected values
		loss = loss_func(y_pred, y_batch)
		# Reset the optimizer's biases and weights
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
	
	# Get MSE (mean squared error) of the training and test parts
	# To see how far away we are from the desired outcome
	with torch.no_grad():
		y_pred = model(x_train)
		train_rmse = np.sqrt(loss_func(y_pred, y_train))
		y_pred = model(x_test)
		test_rmse = np.sqrt(loss_func(y_pred, y_test))
	print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
	



# Format data for plotting

num_future_preds = 5

dates_plot = dates[len(dates) - window_size :]
day = datetime.timedelta(days=1)
norm_plot  = norm[len(norm) - window_size :]
# add more days for future predictions
for i in range(1, num_future_preds+1):
	dates_plot.append( datetime.date(2023, 11, 10) + i*day )
	norm_plot.append(np.nan)
	
# After training + validation, make future predictions
model.eval()
test_size = len(ts) - num_future_preds
predictions = torch.tensor([])
for i in range(test_size):
	seq = torch.FloatTensor(ts[-test_size :])
	
	with torch.no_grad():
		model.hidden = (torch.zeros(1, 1, model.hidden_layers), 
						torch.zeros(1, 1, model.hidden_layers))
		ts.append(model(seq).item())
		test_tensor = torch.tensor(ts[-num_future_preds :])
		predictions = torch.cat( (predictions, test_tensor) )

print(f"len predictions = {len(predictions)}")
print(f"predictions = {predictions}")
print(f"len dates = {len(dates_plot)}")

# Plot data
fig, ax = plt.subplots()
plt.plot(dates_plot, norm_plot, c="b")
plt.plot(dates_plot, predictions, c="r")
ax.scatter(dates_plot, norm_plot, color="blue")
ax.scatter(dates_plot, predictions, color="red")

ax.set_ylim([0, 1])
ax.set_xlabel("Dates")
plt.show()



