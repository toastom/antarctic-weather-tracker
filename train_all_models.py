#!/usr/bin/env/ python3
import os

dtypes = ["tavg", "tmin", "tmax"] #, "prcp", "snwd"]
#window_size = 50
# SELECT COUNT(dt) FROM Days; = 23659
window_size = 23658
num_epochs = 1000
future_preds = 365

for d in dtypes:
	os.system(f"python3 flask/train_model.py {d} {window_size} {num_epochs} {future_preds}")