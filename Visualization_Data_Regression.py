import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#To see all columnsn
pd.set_option('display.max_columns', None)

#Load the data set
auto_prices = pd.read_csv(r'C:\Users\sg_cl\Desktop\Pythonproject\Final exam\Automobile price data _Raw_.csv')
#print(auto_prices.head())
#print(auto_prices.describe())

##################  Clean the Data Set:
# 1) Fix columnnames
# 2) fix the missing values
# 3) Transform the column data type

#print(auto_prices.shape)

auto_prices.columns = [str.replace('-','_') for str in auto_prices.columns]


for col in auto_prices.columns:
    auto_prices.loc[auto_prices[col] == '?', col] = np.nan
#auto_prices.dropna(axis = 0, inplace = True)
auto_prices.fillna(method = 'ffill')


#print(auto_prices.head())

cols = ['normalized_losses','price','peak_rpm','horsepower','stroke']

for col in cols:
    auto_prices[col] = pd.to_numeric(auto_prices[col])
    
#print(auto_prices.dtypes)
    
###############  Exploring te data, FREQUENCY Table

auto_prices['count'] = 1
#print(auto_prices[['make','count']].groupby('make').count())


def count_unique(auto_prices, cols):
    for col in cols:
        print('\n' + 'For column ' + col)
        print(auto_prices[col].value_counts())
        
#all_cols = auto_prices.columns
all_cols =['make','fuel_type','aspiration','num_of_doors','body_style'
           ,'drive_wheels','engine_location','engine_type', 'num_of_cylinders', 
            'fuel_system','price']

#print(count_unique(auto_prices, all_cols))

#############  Visualizing Automobile data for regression

