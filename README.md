# udacity-data-scientist-nanodegree-project-2-disaster-response-pipelines
In this project, I've analyzed disaster data from Figure Eight and built a model for an API that classifies disaster messages.

This project consists of three parts:

## 1. ETL Pipeline
**process_data.py** writes a data cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

## 2. ML Pipeline
**train_classifier.py** write a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

## 3. Flask Web App

* Modify file paths for database and model as needed
* Add data visualizations using Plotly in the web app

## Project File Tree

\- app
\| \- template
\| \|\- master.html  \# main page of web app
\| \|\- go.html  \# classification result page of web app
\|\- run.py  \# Flask file that runs app

\- data
\|\- disaster_categories.csv  \# data to process
\|\- disaster_messages.csv  \# data to process
\|\- process_data.py
\|\- DisasterResponse.db   \# database to save clean data to

\- models
\|\- train_classifier.py
\|\- classifier.pkl  \# saved model

\- README.md \# this file

