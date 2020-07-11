import pandas as pd
from sklearn import datasets
import numpy as np

# import data set from dataset
iris = datasets.load_iris()

#Create Data Frame
species = [iris.target_names[x] for x in iris.target]
iris = pd.DataFrame(iris['data'], columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
iris['Species'] = species

#print(iris)
#print(iris.dtypes)

iris['count'] = 1
print(iris[['Species','count']].groupby('Species').count())

###Plot the features

def plot_iris(iris, col1, col2):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.lmplot(x = col1, y = col2, data = iris, hue = 'Species',fit_reg = False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('Iris species shown by color')
    plt.show()

#plot_iris(iris, 'Petal_Width', 'Sepal_Length')

#plot_iris(iris, 'Petal_Length', 'Sepal_Width')


########## Prepare the dataset

from sklearn.preprocessing import scale
num_cols = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

### Scaling
iris_scaled = scale(iris[num_cols])
iris_scaled = pd.DataFrame(iris_scaled, columns = num_cols)
print(iris_scaled.describe().round(3))

levels = {'setosa':0,
          'versicolor':1,
          'virginica':2}

iris_scaled['Species'] = [levels[x] for x in iris['Species']]

#print(iris_scaled.head())

### Split data in to training and evaluation data set

from sklearn.model_selection import train_test_split
np.random.seed(3456)
iris_split = train_test_split(np.asmatrix(iris_scaled), test_size = 75)
iris_train_features = iris_split[0][:,:4]
iris_train_labels = np.ravel(iris_split[0][:,4])
iris_test_features = iris_split[1][:,:4]
iris_test_labels = np.ravel(iris_split[1][:,4])

print(iris_train_features.shape)
print(iris_train_labels.shape)
print(iris_test_features.shape)
print(iris_test_labels.shape)



#### Train and Evaluate KNN Model

from sklearn.neighbors import KNeighborsClassifier 
KNN_mod = KNeighborsClassifier(n_neighbors = 3)
KNN_mod.fit(iris_train_features, iris_train_labels)

iris_test = pd.DataFrame(iris_test_features, columns  = num_cols)
iris_test['predicted'] = KNN_mod.predict(iris_test_features)
iris_test['correct'] = [1 if x == z else 0 for x,z in zip(iris_test['predicted'], iris_test_labels)]
accuracy = 100 * float(sum(iris_test['correct']))/ float(iris_test.shape[0])
print(accuracy)





