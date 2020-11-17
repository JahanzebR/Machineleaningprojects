import sys
import pandas as pd
import numpy as np
import sklearn
import matplotlib


# print('Python: {}'.format(sys.version))
# print('Pandas:{}'.format(pd.__version__))
# print('NumPy:{}'.format(np.__version__))
# print('Scikit-learn:{}'.format(sklearn.__version__))
# print('MatplotLib:{}'.format(matplotlib.__version__))

from sklearn import datasets

# Load dataset
iris = datasets.load_iris()
features = iris.data
target = iris.target

# generate a Pandas DataFrame
df = pd.DataFrame(features)
df.columns = iris.feature_names

# print(target)
# print(iris.target_names)

# Print dataset informaton

# print(df.shape)
# print(df.head(20))

# Print dataset descriptions and class distributions
# print(df.describe())

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# display scatter plot matrix
# scatter_matrix(df)
# plt.show()

# Elbow method to determine optimal number of clusters

from sklearn.cluster import KMeans

# empty x and y data lists
X = []
Y = []


for i in range(1, 31):
    # initialize and fit the kmeanns model
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)

    #  append the number of clusters to x data list
    X.append(i)

    # append average within-cluster sum of squares to y data list
    awcss = kmeans.inertia_ / df.shape[0]
    Y.append(awcss)

import matplotlib.pyplot as plt

# plot the x and y data
plt.plot(X, Y, 'bo-')
plt.xlim(1, 30)
plt.xlabel('Number of Clusters')
plt.ylabel('Average Within-Cluster Sum of Squares')
plt.title('K-Means Clustering Elbow Method')

# display the plot
# plt.show()

# * Principle Component Analysis
# From Wikipedia - principal component analysis (PCA) is a statistical procedure that
# uses an orthogonal transformation to convert a set of observations of possibly correlated
# variables into a set of values of linearly uncorrelated variables called principal components.

# Unless you have a heavy linear algebra background, it's easy to get lost in the eigenvectors and eigenvalues that PCA relies on.
# Scikit-learn makes implementing PCA easy and straight forward.
# I perform a PCA reduction to reduce the number of features in our dataset to two.
# As a result of this reduction, we will be able to visualize each instance as an X,Y data point.

from sklearn.decomposition import PCA
from sklearn import preprocessing

# perform PCA
pca = PCA(n_components=2)
pc = pca.fit_transform(df)

# print new dimension
# print(pc.shape)
# print(pc[:10])

# re-fit kmeans model to the principle components with the appropriate number of clusters

kmeans = KMeans(n_clusters=3)
kmeans.fit(pc)
params = kmeans.get_params()
# print(params)

# Visualize high dimensional clusters using principle components

# set size for the mesh
h = 0.02

# generate mesh grid
x_min, x_max = pc[:, 0].min() - 1, pc[:, 0].max() + 1
y_min, y_max = pc[:, 1].min() - 1, pc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# label each point in mesh using last trained model
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# generate color plot from results
Z = Z.reshape(xx.shape)
plt.figure(figsize=(12, 12))
plt.clf()
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.tab20c, aspect="auto", origin='lower')

# plot the principal components on the color plot
for i, point in enumerate(pc):
    if target[i] == 0:
        plt.plot(point[0], point[1], 'g.', markersize=10)
    if target[i] == 1:
        plt.plot(point[0], point[1], 'r.', markersize=10)
    if target[i] == 2:
        plt.plot(point[0], point[1], 'b.', markersize=10)

# plot the cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=250, linewidth=4, color='w', zorder=10)

# set plot title and axis limits
plt.title('K-Means Clustering on PCA-reduced Iris Data Set')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xticks(())
plt.yticks(())

# display
# plt.show()

# * Clustering Metrics
# It looks good! But did the PCA reduction impact the performance of the
# Kmeans clustering algorithm? Investigate by using some common clustering metrics, such as homogeneity, completeness, and V-measure.

# * Homogeneity - measures whether or not all of its clusters contain only
#   data points which are members of a single class.
# * Completeness - measures whether or not all members of a given class are
#   elements of the same cluster
# * V-measure - the harmonic mean between homogeneity and completeness

from sklearn import metrics

# K Means clustering on non reduced data
kmeans1 = KMeans(n_clusters=3)
kmeans1.fit(features)

# K Means Clustering on PCA reduced data
kmeans2 = KMeans(n_clusters=3)
kmeans2.fit(pc)

# print metrics for non reduced data
print("Non-reduced data")
print("Homogeneity: {}".format(metrics.homogeneity_score(target, kmeans1.labels_)))
print('Completeness: {}'.format(metrics.completeness_score(target, kmeans1.labels_)))
print('V-measure: {}'.format(metrics.v_measure_score(target, kmeans1.labels_)))

# print metrics for PCA reduced data
print("PCA Reduced data")
print("Homogeneity: {}".format(metrics.homogeneity_score(target, kmeans2.labels_)))
print('Completeness: {}'.format(metrics.completeness_score(target, kmeans2.labels_)))
print('V-measure: {}'.format(metrics.v_measure_score(target, kmeans2.labels_)))

# compare results further, print out the actual labels
print(kmeans1.labels_)
print(kmeans2.labels_)
print(target)
