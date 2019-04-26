# from sklearn.datasets import load_iris
# from sklearn.cluster import KMeans
#
# print(KMeans)
# data = load_iris()
# cluster = KMeans(n_clusters=3, random_state=None, init_method='k-means++', n_init=3,)

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import math as math

# Get the data of iris
iris = load_iris()

# Get Sepal Length and save them in x1
x1 = iris.data[:, :1]
# Get target value and save them in x_target
x_target = iris.target
x2_list = []
x2_List = []
x3_list = []
x3_List = []

# Get Sepal Width and save them in x2
# Compute squared value of the sum between the quadratic value of sepal length and the quadratic value of sepal width
for i in range(0, len(x1)):
    x2_list.append(iris.data[i][1])
    x2_List.append(x2_list)
    x2_list = []
    x3_list.append(math.sqrt((iris.data[i][0])**2+(iris.data[i][1])**2))
    x3_List.append(x3_list)
    x3_list = []

# Save Sepal Width value in x2
x2 = np.array(x2_List)
# Save the quadratic value of sepal length and the quadratic value of sepal width in x3
x3 = np.array(x3_List)


# create new plot and data
plt.plot()
X = np.array(list(zip(x3, x_target))).reshape(len(x3), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1, 150)

# Fit x3 and x_target in the for loop
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method')
plt.show()

