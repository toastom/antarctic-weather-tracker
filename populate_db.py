#!/bin/usr/env python3

import json
import time

"""
Iterate through all year.json files and each response within
Collect all values for each day in a day dictionary w/ primary key = date
Generate SQL code to populate the database's Days table and save in a separate SQL file

"""

class Day:
	def __init__(self, date=None, tmin=None, tmax=None, tavg=None, prcp=None, snwd=None):
		self.date = date
		self.tmin = tmin
		self.tmax = tmax
		self.tavg = tavg
		self.prcp = prcp
		self.snwd = snwd

# Organize all results of a single day into a single dictionary entry
def organize_days(response_list, prev_dict=None):
	days = {}
	if(prev_dict != None):
		days = prev_dict
	
	for resp in response_list:
		# Clean up the date str
		date = resp["date"][:10]
		
		# Check if this date has already been added to the days dict
		if(date not in days):
			
			new_day = None
			val = resp["value"]
			
			if(resp["datatype"] == "TMIN"):
				new_day = Day(date=date, tmin=val)
				
			elif(resp["datatype"] == "TMAX"):
				new_day = Day(date=date, tmax=val)
				
			elif(resp["datatype"] == "TAVG"):
				new_day = Day(date=date, tavg=val)
				
			elif(resp["datatype"] == "PRCP"):
				new_day = Day(date=date, prcp=val)
				
			elif(resp["datatype"] == "SNWD"):
				new_day = Day(date=date, snwd=val)
				
			days.update( { date : new_day } )
			
		# If it has been, then just update the values with whatever this new value is
		else:
			val = resp["value"]
			days.get(date).date = date
			
			if(resp["datatype"] == "TMIN"):
				days.get(date).tmin = val
				
			elif(resp["datatype"] == "TMAX"):
				days.get(date).tmax = val
				
			elif(resp["datatype"] == "TAVG"):
				days.get(date).tavg = val
				
			elif(resp["datatype"] == "PRCP"):
				days.get(date).prcp = val
				
			elif(resp["datatype"] == "SNWD"):
				days.get(date).snwd = val
					
	return days


# Generate SQL code to insert values into the database based on the dict returned from organize_days()
def generate_sql(result, year):
	# For each dictionary date key, generate SQL code to insert the values into a new row of table Days
	
	code = open("generated_sql.sql", "a+")
	code.write(f"-- {year}\n")
	#last_post = code.tell()
	code.write("INSERT INTO Days(dt, tavg, tmin, tmax, prcp, snwd) VALUES\n")
	
	keys_list = result.keys()
	for k in keys_list:
		tavg = result.get(k).tavg if (result.get(k).tavg != None) else "NULL"
		tmin = result.get(k).tmin if (result.get(k).tmin != None) else "NULL"
		tmax = result.get(k).tmax if (result.get(k).tmax != None) else "NULL"
		precip = result.get(k).prcp if (result.get(k).prcp != None) else "NULL"
		snowd  = result.get(k).snwd if (result.get(k).snwd != None) else "NULL"
		
		ch = ""
		klist = list(keys_list)
		if(klist.index(k) == len(klist)-1):
			ch = ";"
		else:
			ch = ","
		code_str = f"('{k}', {tavg}, {tmin}, {tmax}, {precip}, {snowd}){ch}\n"
		code.write(code_str)
	
	code.write('\n')
	code.close()

# Out of one of the two API responses in a year.json file
# full results list, individual day's datatype + results dict, single value from that dict
# response["results"][0]["datatype"]

def main():
	
	code = open("generated_sql.sql", "w")
	code.write("".join((
	"CREATE DATABASE IF NOT EXISTS weather_database;\n",
	"USE weather_database;\n\n",
	"CREATE TABLE IF NOT EXISTS Days(\n",
		"\tdt Date NOT NULL,\n",
		"\ttavg float,\n",
		"\ttmin float,\n",
		"\ttmax float,\n",
		"\tprcp float,\n",
		"\tsnwd float,\n",
		"\tPRIMARY KEY(dt)\n",
	");\n\n"
	)))
	code.close()
	
	log_file = open("popdb_log.txt", "a")
	
	for year in range(1957, 2024):
		file = open(f"data_files/{year}.json")
		file_str = file.read()
		file.close()
		
		# Separate the two responses stored in this file out into each response's own json object dict
		first_response = json.JSONDecoder().raw_decode(file_str)
		second_response = json.JSONDecoder().raw_decode(file_str[first_response[1] + 1 : ])

		json_list1 = first_response[0].get("results")
		json_list2 = second_response[0].get("results")
		
		try:
			# Sometimes the second response is empty, and if so, make sure to skip processing its results
			days1 = organize_days(json_list1)
			#generate_sql(days1, year)
		
			# Collection of all data in all days this year between both the first and second request
			# This is a fix for a bug that happened when one day's data was cut between the first and second response
			# causing a duplicate day when generating the SQL code
			all_days = None
			if(json_list2 != None):
				all_days = organize_days(json_list2, days1)
		
		
			if(all_days != None):
				generate_sql(all_days, year)
			else:
				generate_sql(days1, year)
				
				
		except Exception as err:
			log_file.write(time.asctime() + f":\t Year: {year}\n")
			log_file.write(str(type(err)) + ": " + str(err) + "\n")
			print(time.asctime() + f": Year: {year}", end="\t")
			print(str(type(err)) + ": " + str(err) + "\n")
		
	log_file.close()
	
main()