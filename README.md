## Antarctic Weather Tracker

This is a project to download, store, organize, and plot Antarctic weather data gathered from the NOAA National Centers for Environmental Information ([NOAA NCEI](https://ncei.noaa.org/)). It started as my final project for Dr. Tong's Fall 2023 Database Systems class at Georgia Southern University but is now being expanded beyond its initial scope.  

Information about the initial project planning and overview can be found in the Final Report PDF in this repository. This includes database design, ER diagram, motivations, and challenges encountered during the initial design and implementation of the project as of November 2023.

### Project Roadmap  

In no particular order:  
- [ ] Add more Antarctic weather stations besides just Base Orcadas
- [ ] Implement advanced search functionality
	- [ ] Search based on weather station
	- [ ] Filter hottest/coldest/snowiest days of the selected date range
	- [ ] Return all data from the user's query (csv file for larger queries?) in addition to plotting over the user's selected timeframe
- [ ] SQLi protection and user input sanitation
- [ ] Regularly and automatically update the database as new NOAA data becomes available
- [ ] General UI cleanup
- [ ] Implement AI weather prediction based on decades of past weather trends
	- [ ] With enough weather stations, project this to make a generalization for Antarctic weather as a whole


### Development

For those who want to help with development, this project is just a Python Flask app that uses simple HTML/CSS templates as Flask endpoints. Everything to do with the UI and web app to interact with the `weather_database.sql` can be found in the `/flask` folder. The `/data_files` are the initial files created to store the all the original Base Orcadas weather data, with each file holding the raw response data from a request to the NCEI API. Finally, the rest of the code found in the root directory, namely `download_data.py` and `populate_db.py`, were helper scripts made to automate the download process of tens of thousands of datapoints, organize them to fit the DB design, and populate the DB. Once again, a better explanation of the initial idea can be found in the `Final Report.pdf`.

To run the app, simply `git clone` and `cd flask`, then run `flask --app app run`. You will of course need Python3, Flask, and git installed to do this.


