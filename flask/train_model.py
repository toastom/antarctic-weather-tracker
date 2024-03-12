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

def denormalize(data_points, mini, maxi, dtype="tmin"):
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


q = Query()
q.send_query("SELECT dt, tmin FROM Days")
full_dataset = q.get_result_cols()
dates = full_dataset[0]
tmin  = full_dataset[1]

norm = normalize(tmin)

window_size = 2000
tr, ts = create_sets(norm, window_size)
print(f"Window size: {window_size}")
#print(f"Training set: {tr}")
#print(f"Test set: {ts}")

# x = feature; y = target
lookback = 1
x_train, y_train = create_tensors(tr, lookback)
x_test, y_test = create_tensors(ts, lookback)
print("x_train shape: ", x_train.shape, "\ty_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape, "\ty_test shape: ", y_test.shape)



# We have the dataset, now create the model
model = WeatherModel()
optimizer = optim.Adam(model.parameters())
# Loss function for calculating error and to train to minimize error
loss_func = nn.MSELoss() # MSE = Mean Square Error

# Batch size must = 1 right now or else we get a dimension error
# Not sure if shuffle should = True or False
loader = td.DataLoader(td.TensorDataset(x_train, y_train), shuffle=True, batch_size=1)

# Train the dataset

num_epochs = 1200
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
	

# Format data for plotting

dates_plot = dates[len(dates) - window_size :]
day = datetime.timedelta(days=1)
norm_plot  = norm[len(norm) - window_size :]

future_preds = 25
# add more days for future predictions
for i in range(1, future_preds+1):
	dates_plot.append( datetime.date(2023, 11, 10) + i*day )
	norm_plot.append(np.nan)
	
# After training + validation, make future predictions
model.eval()

predictions = torch.tensor([])
print("predictions shape: ", predictions.shape)

print("y_pred shape: ", y_pred.shape)
print("y_pred: ", y_pred)

with torch.no_grad():
	train_tens = torch.FloatTensor(tr)
	test_tens  = torch.FloatTensor(ts)
	
	model_train = model(train_tens)
	model_test  = model(test_tens)
	predictions = torch.cat((predictions, model_train), dim=0)
	predictions = torch.cat((predictions, model_test),  dim=0)

	
	# Plot future dates now
	# minus 2 because the model already predicts 2 future dates by default
	# Go off the last prediction to plot the next
	#for i in range(future_preds-2):
	pred_tens = torch.FloatTensor(predictions[-future_preds+2:])
	predictions = torch.cat((predictions, model(pred_tens)), dim=0)

new_tmin = impute(tmin)
temp_plot = denormalize(norm_plot, min(new_tmin), max(new_tmin))
denorm_preds = denormalize(predictions.numpy(), min(new_tmin), max(new_tmin))

print(f"shape predictions = {predictions.shape}")
print(f"len dates = {len(dates_plot)}")

torch.save(model.state_dict(), "../weather_model.pt")

# Plot data
fig, ax = plt.subplots()
plt.plot(dates_plot, temp_plot, c="g")
plt.plot(dates_plot, denorm_preds, c="r")
ax.scatter(dates_plot, temp_plot, color="green")
ax.scatter(dates_plot, denorm_preds, color="red")
ax.set_ylim([min(temp_plot), max(temp_plot)])

# Plot data
fig, ax = plt.subplots()
plt.plot(dates_plot, norm_plot, c="b")
plt.plot(dates_plot, predictions, c="r")
ax.scatter(dates_plot, norm_plot, color="blue")
ax.scatter(dates_plot, predictions, color="red")

ax.set_ylim([0, 1])
ax.set_xlabel("Dates")
plt.show()



