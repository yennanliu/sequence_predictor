# python 3 



import datetime
import pandas as pd 



def timestamp_fix_(x):
    # fix mm/dd/yy -> yyyy-mm-dd
    y_=x.split('/')
    mm,dd,yy = y_[0], y_[1],y_[2]
    yy = '20'+yy
    y_output = yy+'/'+ mm +'/'+ dd
    return y_output