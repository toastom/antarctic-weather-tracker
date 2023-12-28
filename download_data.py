#!/bin/env/python3

"""
 Make requests to the NOAA Weather API to get the daily summaries of
 the weather station at Base Orcadas, Antarctica.
 Return average temperature, max temp, min temp, precipitation and snow depth data

 GHCND is the dataset for daily summmaries, can only read for one year at a time
 (limit to 5*366 = 1830 entries because each entry is a separate unit avg temp, min temp, etc)
 (max limit is 1000 though so we must do two requests per year with offset = 1001 for the second request and to prevent overlap of data)

"""

import json
import requests
import math
import time


url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
init_year = 1957
start_date = f"{init_year}-01-01"
end_date   = f"{init_year}-12-31"
args = { "datasetid" : "GHCND", "datatypeid" : "TAVG,TMAX,TMIN,PRCP,SNWD", "stationid" : "GHCND:AYM00088968",
		"units" : "standard", "startdate" : start_date, "enddate" : end_date, "limit" : "1000", "offset" : "0" }
# Retrieve your own token from the NCEI website
headers = {"token" : "<your-token_here>"}

def save_weather_data(year, elapsed_time, req_count, prev_count):
	args["startdate"] = f"{year}-01-01"
	args["enddate"]   = f"{year}-12-31"
	args["offset"] = "0"
	req1 = requests.get(url, params=args, headers=headers)

	# Sleep 1 sec to prevent more than 5 requests from ever happening in 1 sec
	time.sleep(1)

	args["offset"] = "1001"
	req2 = requests.get(url, params=args, headers=headers)

	# Prettify and convert to str
	req1_pretty = json.dumps(req1.json(), sort_keys=True, indent=4)
	req2_pretty = json.dumps(req2.json(), sort_keys=True, indent=4)

	file = open(f"data_files/{year}.json", "w")
	file.write(req1_pretty + "\n")
	file.write(req2_pretty + "\n")
	file.close()

	return (req1, req2, req_count, prev_count)

if(__name__ == "__main__"):
	print("Starting download...")

	start_time = time.time()
	log_file = open("log.txt", "a")
	log_file.write(time.asctime() + "\n")

	req_count  = 0
	prev_count = 0
	prev_time = time.time()
	for year in range(init_year, 2024):
		resp = None
		elapsed_time = time.time() - prev_time
		try:
			prev_time = time.time()
			resp = save_weather_data(year, elapsed_time, req_count, prev_count)
			req_count  = resp[2]
			prev_count = resp[3]
			
			"""
			Kinda unnecessary since this script will only make ~100 requests
			if(req_count >= 10000):
				log_file.write("ERROR: \# requests at {} and over 10000 today.\nExiting...\n\n".format(req_count))
				log_file.close()
				break
			"""
			log_file.write(f"Saved data_files/{year}.json\n")
			print(f"Saved data_files/{year}.json")
		except Exception as err:
			log_file.write(str(type(err)) + ": " + str(err) + "\n")
			log_file.write("API response, \# reqs: " + str(resp) + "\n")
			log_file.write(f"ERROR: Failed to write data_files/{year}.json\n")

	total_time = time.time() - start_time
	secs  = total_time % 60
	mins  = math.floor((total_time % 3600) / 60)
	hours = math.floor(total_time / 3600)

	print("Download complete!")
	print("Total time: {}:{}:{}".format(hours, mins, round(secs, 2)))
	log_file.write("Download complete!\n")
	log_file.write("Elapsed time: {}:{}:{}\n\n".format(hours, mins, round(secs, 2)))
	log_file.close()
