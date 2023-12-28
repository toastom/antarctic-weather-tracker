#!/usr/bin/env python3

# query.py
# Wrapper to simplify SQL query creation and accessing the database from the main Flask app

import mysql.connector

class Query():
	def __init__(self):
		self.query  = None
		self.result = None
		
	def construct_query(self, col1="*", col2="", table="Days", where=False, cond="", like=False, patt="", btw=False, start="", end=""):
		if(col2):
			col1 = f"{col1}, {col2}"
			col2 = ""
		
		if(where):
			where = "WHERE "
			if(cond == ""):
				raise ValueError("Condition 'cond' should not be empty when 'where' is True")
		else:
			where = ""
		
		if(like):
			like = "LIKE "
			if(patt == ""):
				raise ValueError("Pattern 'patt' should not be empty when 'like' is True")
		else:
			like = ""
		
		if(btw and start and end):
			btw = f"BETWEEN '{start}' AND '{end}'"
		else:
			btw = ""
			
		
		query = f"SELECT {col1} FROM {table} {where} {cond} {like} {patt} {btw}"
		self.query = query
		return query
	
	def send_query(self, query=None):
		weather_db = mysql.connector.connect(host="localhost", port="3306", user="user1", password="", database="weather_database")
		db = weather_db.cursor()
		#self.query = None
		
		#query = "SELECT dt, tmin FROM Days WHERE YEAR(dt) = '2001'"
		if(self.query):
			db.execute(self.query)
		elif(query):
			db.execute(query)
		else:
			raise ValueError("Query class member 'self.query' undefined")
		
		res = db.fetchall()
		self.result = res
		db.close()
		
		return res
	
	def get_result_cols(self):
		if(self.result == None):
			raise TypeError("self.result must not be None. Did you call send_query() first?")
		
		# Loop through columns by rows
		res = []
		for c in range(0, len(self.result[0])):
			col = []
			for row in self.result:
				col.append(row[c])
			
			res.append(col)
		
		# Should prob replace the above with the simpler way if we're only searching for two columns here
		"""
		for row in res:
			days.append(row[0])
			tmin.append(row[1])
			#print(row)
		"""
		
		return res
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	