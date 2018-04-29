

# ref 
# https://github.com/neelabhpant/Deep-Learning-in-Python/blob/master/Time%20Series%20Prediction.ipynb


# ops 
import pandas as pd 
import numpy as np 
# ML
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# help function 
#---------------------------------------

# data prepare 
def get_data(fine_name):
	print (fine_name)
	df = pd.read_csv('data/{}.csv'.format(fine_name))
	print (df.head())
	return df  

def col_fix(df):
    df = df.drop('Unnamed: 0', axis=1) 
    print (df.columns)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    print (df.head())
    return df 

def train_test_split(df,col):
	df_ = df.set_index('Date')
	split_date  = pd.Timestamp('2016-01-01')
	train = df_.loc[:split_date]
	test = df_.loc[split_date:]
	sc = MinMaxScaler()
	train_sc = sc.fit_transform(train)
	test_sc = sc.transform(test)
	train_sc_df = pd.DataFrame(train_sc, columns=[col], index=train.index)
	test_sc_df = pd.DataFrame(test_sc, columns=[col], index=test.index)

	for s in range(1,2):
		train_sc_df['X_{}'.format(s)] = train_sc_df[col].shift(s)
		test_sc_df['X_{}'.format(s)] = test_sc_df[col].shift(s)
	
	X_train = train_sc_df.dropna().drop([col], axis=1)
	y_train = train_sc_df.dropna().drop('X_1', axis=1)

	X_test = test_sc_df.dropna().drop([col], axis=1)
	y_test = test_sc_df.dropna().drop('X_1', axis=1)

	X_train = X_train.as_matrix()
	y_train = y_train.as_matrix()

	X_test = X_test.as_matrix()
	y_test = y_test.as_matrix()

	return train, test,X_train,y_train,X_test, y_test


def SVR_model(X_train,y_train):
	regressor = SVR(kernel='rbf')
	regressor.fit(X_train, y_train)
	y_pred = regressor.predict(X_test)
	return y_pred


def r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))




#---------------------------------------



if __name__ == '__main__':
	df_FB = get_data('FB')
	df_FB_ = col_fix(df_FB)
	train, test,X_train,y_train,X_test, y_test =  train_test_split(df_FB_,'Open')
	y_pred = SVR_model(X_train,y_train)
	r2_test = r2_score(y_test, y_pred)
	print(r2_test)




