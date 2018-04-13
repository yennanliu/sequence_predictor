# python 3 



import datetime
import pandas as pd 
import os


def load_data(filename):
	cwd = os.getcwd()
	route = cwd + '/data/' + filename + '.csv'
	print (route)
	df = pd.read_csv(route)
	print (df.head())
	return df 