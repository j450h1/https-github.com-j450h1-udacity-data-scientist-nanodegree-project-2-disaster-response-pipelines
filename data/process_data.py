# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

# Usage while in this folder:
# python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

def main():
	"""
	Load data, merge it, clean it, and then export it to database
	"""
	# get current directory
	#import os; print(os.getcwd())
	#print(sys.argv[:])

	########################
	# Loads the messages and categories datasets
	########################

	# load messages dataset
	#messages = pd.read_csv('messages.csv')
	messages = pd.read_csv(f'{sys.argv[1]}')

	# load categories dataset
	#categories = pd.read_csv('categories.csv')
	categories = pd.read_csv(f'{sys.argv[2]}')

	#########################
	# Merges the two datasets
	#########################

	# merge datasets
	df = messages.merge(categories) #implicitly on id

	#########################
	# Cleans the data
	#########################
	# Split `categories` into separate category columns.

	# create a dataframe of the 36 individual category columns
	categories = df['categories'].str.split(pat=';', n=36, expand=True)

	# select the first row of the categories dataframe
	row = categories.iloc[0, : ]

	# use this row to extract a list of new column names for categories.
	# one way is to apply a lambda function that takes everything
	# up to the second to last character of each string with slicing
	category_colnames = [name[:-2] for name in row]

	# rename the columns of `categories`
	categories.columns = category_colnames

	#Convert category values to just numbers 0 or 1.
	for column in categories:
	    # set each value to be the last character of the string
	    categories[column] = categories[column].astype(str).str[-1:]
	    # convert column from string to numeric
	    categories[column] = categories[column].astype(int)

	# Replace categories column in df with new category columns.

	# drop the original categories column from `df`
	df.drop(columns = 'categories', inplace = True)

	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.merge(df, categories, left_index=True, right_index=True)

	# drop duplicates
	df.drop_duplicates(inplace=True)

	#########################
	# Stores it in a SQLite database
	#########################

	#engine = create_engine('sqlite:///DisasterResponse.db')
	engine = create_engine(f'sqlite:///{sys.argv[3]}')
	df.to_sql('disaster_table', engine, index=False, 
	if_exists='replace') # Overwrite if table already exists

	print(f"{sys.argv[3]} database created!")

if __name__ == '__main__':
    main()


