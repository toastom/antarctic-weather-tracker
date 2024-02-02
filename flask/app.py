#!/usr/bin/env python3

# After `pip3 install flask` run this with the command `flask --app app run`

from flask import Flask
from flask import request, url_for, render_template, redirect
from markupsafe import Markup
import matplotlib.pyplot as plt
import datetime

# this is just a way to standardize the construction of queries to the DB
import query as q

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("login.html")
	
@app.route('/login', methods=['POST', 'GET'])
def login():
	error = None
	
	if(request.method == 'POST'):
		# Get all valid Users from db
		my_q = q.Query()
		query = my_q.construct_query(col1="userID", col2="name", table="Users")
		users = my_q.send_query()
		
		# Get all valid Admins from db
		# SELECT Admins.userId, Admins.name FROM Admins INNER JOIN Users ON Admins.userId = Users.userId;
		print(request.get_data())
		
		# If checked login as admin
		if(request.form.get("admin") == "True"):
			admin_q = q.Query()
			# Fix this later to actually make use of the Query class
			admin_query = "SELECT Admins.userId, Admins.name FROM Admins INNER JOIN Users ON Admins.userId = Users.userId"
			admins = admin_q.send_query(admin_query)
			
			print(admins)
			#print((request.form['userID'], request.form['username']))
			
			# If user is logging in as admin, check to see if they are an admin
			if( ( int(request.form.get('userID')), request.form.get('username') ) in admins ):
				return redirect(url_for("admin"))
			# Else return admin login error
			error = "User ID / name is not an admin"
			return render_template('login.html', error=error)
		
		
		# If logging in as normal user, then just redirect to the search page
		for row in users:
			userId = row[0]
			name = row[1]

			if(request.form.get('userID') == str(userId) and request.form.get('username') == str(name)):
				error = None
				return redirect(url_for("search"))
			else:
				error = 'Invalid userID/username'
		

	# Else then GET the login form page
	return render_template('login.html', error=error)


@app.route("/search", methods=["POST", "GET"])
def search():
	error = None
	
	if(request.method == 'POST'):
		#print(request)
		#print(request.form)
		
		start = request.form['startDate']
		end = request.form['endDate']
		dtype = request.form['datatype']
		
		my_q = q.Query()
		query = my_q.construct_query(col1="dt", col2=f"{dtype}", where=True, cond="dt", btw=True, start=start, end=end)
		results = my_q.send_query()
		
		# Maybe do some error checking to make sure we have a good reponse from the DB?
		
		# Assuming we only searched for two columns
		# Column 0 is date, column 1 is whatever data we searched
		(dates, data) = my_q.get_result_cols()
		
		# Then generate the plot for the page using (dates, data) and return the file
		fname = generate_plot(start, end, dtype, dates, data)
		
		filepath = url_for("static", filename=fname)
		
		# Now pass the file to the search.html page template
		# search.html template handles the img display

		return render_template("search.html", error=error, datatype=dtype, start=start, end=end, fp=filepath)
		
	
	# the code below is executed if the request method
	# was initially GET or the credentials were invalid
	return render_template("search.html", error=None, datatype=None, start=None, end=None)


@app.route("/admin", methods=["POST", "GET"])
def admin():
	error = None
	
	return render_template("admin.html", error=error)


def generate_plot(start, end, datatype, dates, data):
	fig, ax = plt.subplots()
	ax.scatter(dates, data)
	ax.set_xlabel("Dates")
	
	if(datatype == "tmin" or datatype == "tmax" or datatype == "tavg"):
		ax.set_ylabel(f"{datatype} (*F)")
	else:
		ax.set_ylabel(f"{datatype} (in.)")
	
	ax.set_title(f"{datatype} per day between {start} and {end}")
	#plt.show()
	plt.savefig("static/plot.png")
	plt.close(fig)
	
	# Return the filename to use in the template
	return "plot.png"










