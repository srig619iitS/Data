import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


auto_prices = pd.read_csv(r'C:\Users\sg_cl\Desktop\ML\Automobile price data _Raw_.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#print(auto_prices)


#Fix column names

cols = auto_prices.columns

auto_prices.columns = [str.replace('-','_') for str in cols]

#print(auto_prices.columns)


#Treating missing values

cols = ['price', 'bore', 'stroke', 
          'horsepower', 'peak_rpm']

for column in cols:
    auto_prices.loc[auto_prices[column]=='?', column] = np.nan
auto_prices.dropna(axis = 0, inplace = True)
                   
#print(auto_prices.dtypes) 

#convert somne columns to numeric values

for column in cols:
    auto_prices[column] = pd.to_numeric(auto_prices[column])

#print(auto_prices.dtypes)
#print(auto_prices.describe())



#Create Frequency Table:

def count_unique(auto_prices,cols):
    for col in cols:
        print('\n'+'For Column'+ col)
        print(auto_prices[col].value_counts())

cat_cols = ['make', 'fuel_type', 'aspiration', 'num_of_doors', 'body_style', 
            'drive_wheels', 'engine_location', 'engine_type', 'num_of_cylinders', 
            'fuel_system']

#print(count_unique(auto_prices, cat_cols))

#1-d plots
from mpl_toolkits.mplot3d import Axes3D
#Plotting Bar graphs for Frequency Table
def plot_bars(auto_prices,cols):
    for col in cols:
        fig = plt.figure(figsize = (6,6))
        ax = fig.gca(projection="3d")
        counts = auto_prices[col].value_counts()
        counts.plot.bar(ax = ax, color = 'blue')
        plt.show()
    
plot_cols = ['make', 'body_style', 'num_of_cylinders']
#plot_bars(auto_prices, plot_cols)

#2-d plots

def plot_scatter(auto_prices, cols, col_y = 'price'):
    for col in cols:
        fig = plt.figure(figsize = (7,6))
        ax = fig.gca()
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax)
        plt.show()

num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']
#plot_scatter(auto_prices, num_cols)        

#plot_scatter(auto_prices, ['horsepower'], 'engine_size')

#adding transparency arguement

def plot_scatter_t(auto_prices, cols, col_y = 'price',alpha = 1.0):
    for col in cols:
        fig = plt.figure(figsize = (7,6))
        ax = fig.gca()
        auto_prices.plot.scatter(x = col, y = col_y, ax = ax, alpha = alpha)
        plt.show()

num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']
#plot_scatter(auto_prices, num_cols)   

#plot_scatter_t(auto_prices, num_cols, alpha = 0.2)        

# 1 D KDE alongh with Countour plots
def plot_desity_2d(auto_prices, cols, col_y = 'price', kind ='kde'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.jointplot(col, col_y, data=auto_prices, kind=kind)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()

#plot_desity_2d(auto_prices, num_cols)      

#plot_desity_2d(auto_prices, num_cols, kind = 'hex')   

#Box plot
def plot_box(auto_prices, cols, col_y = 'price'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col, col_y, data=auto_prices)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()
        
cat_cols = ['fuel_type', 'aspiration', 'num_of_doors', 'body_style', 
            'drive_wheels', 'engine_location', 'engine_type', 'num_of_cylinders'
            ]
plot_box(auto_prices, cat_cols)    

#violen plot
def plot_violin(auto_prices, cols, col_y = 'price'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=auto_prices)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()
        
#plot_violin(auto_prices, cat_cols)

#Marker shape

def plot_scatter_shape(auto_prices, cols, shape_col = 'fuel_type', col_y = 'price', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^']
    unique_cats = auto_prices[shape_col].unique()
    #print(unique_cats)
    for col in cols:
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats):
            temp = auto_prices[auto_prices[shape_col] == cat]
            #print(temp.head())
            sns.regplot(col,col_y,data = temp, marker = shapes[i], label = cat,scatter_kws = {"alpha":alpha}, fit_reg = False, color = 'blue')
            plt.title("SS")
            plt.xlabel(col)
            plt.ylabel(col_y)
            plt.legend()
            plt.show()
            
num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']
#plot_scatter_shape(auto_prices, num_cols)
            
 
#Size 
    
def plot_scatter_size(auto_prices, cols, shape_col = 'fuel_type', size_col = 'curb_weight',
                            size_mul = 0.000025, col_y = 'price', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] # pick distinctive shapes
    unique_cats = auto_prices[shape_col].unique()
    for col in cols: # loop over the columns to plot
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): # loop over the unique categories
            temp = auto_prices[auto_prices[shape_col] == cat]
            sns.regplot(col, col_y, data=temp, marker = shapes[i], label = cat,
                        scatter_kws={"alpha":alpha, "s":size_mul*auto_prices[size_col]**2}, 
                        fit_reg = False, color = 'blue')
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.legend()
        plt.show()

num_cols = ['engine_size', 'horsepower', 'city_mpg']
#plot_scatter_size(auto_prices, num_cols)  

#Color & Size

def plot_scatter_shape_size_col(auto_prices, cols, shape_col = 'fuel_type', size_col = 'curb_weight',
                            size_mul = 0.000025, color_col = 'aspiration', col_y = 'price', alpha = 0.2):
    shapes = ['+', 'o', 's', 'x', '^'] # pick distinctive shapes
    colors = ['green', 'blue', 'orange', 'magenta', 'gray'] # specify distinctive colors
    unique_cats = auto_prices[shape_col].unique()
    unique_colors = auto_prices[color_col].unique()
    for col in cols: # loop over the columns to plot
        sns.set_style("whitegrid")
        for i, cat in enumerate(unique_cats): # loop over the unique categories
            for j, color in enumerate(unique_colors):
                temp = auto_prices[(auto_prices[shape_col] == cat) & (auto_prices[color_col] == color)]
                sns.regplot(col, col_y, data=temp, marker = shapes[i],
                            scatter_kws={"alpha":alpha, "s":size_mul*temp[size_col]**2}, 
                            label = (cat + ' and ' + color), fit_reg = False, color = colors[j])
        plt.title('Scatter plot of ' + col_y + ' vs. ' + col) # Give the plot a main title
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.legend()
        plt.show()

num_cols = ['engine_size', 'horsepower', 'city_mpg']        
#plot_scatter_shape_size_col(auto_prices, num_cols)   

#split violin plots

def plot_violin_hue(auto_prices, cols, col_y = 'price', hue_col = 'aspiration'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col, col_y, data=auto_prices, hue = hue_col, split = True)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()
        
plot_violin_hue(auto_prices, cat_cols)    
#print(cat_cols)

#pairplot Multi dimensional view
num_cols = ["curb_weight", "engine_size", "horsepower", "city_mpg", "price", "fuel_type"] 
#sns.pairplot(auto_prices[num_cols], hue='fuel_type', palette="Set2", diag_kind="kde", size=2).map_upper(sns.kdeplot, cmap="Blues_d")

#Conditioned Histograms

def cond_hists(df, plot_cols, grid_col):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ## Loop over the list of columns
    for col in plot_cols:
        grid1 = sns.FacetGrid(df, col=grid_col)
        grid1.map(plt.hist, col, alpha=.7)
    return grid_col

## Define columns for making a conditioned histogram
plot_cols2 = ["length",
               "curb_weight",
               "engine_size",
               "city_mpg",
               "price"]

cond_hists(auto_prices, plot_cols2, 'drive_wheels')

# aConditioned plot
def cond_plot(cols):
    import IPython.html.widgets
    import seaborn as sns
    for col in cols:
        g = sns.FacetGrid(auto_prices, col="drive_wheels", row = 'body_style', 
                      hue="fuel_type", palette="Set2", margin_titles=True)
        g.map(sns.regplot, col, "price", fit_reg = False)

num_cols = ["curb_weight", "engine_size", "city_mpg"]
#cond_plot(num_cols)  














