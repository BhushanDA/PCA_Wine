import pandas as pd 
import numpy as np
win = pd.read_csv(r"D:\Python\wine.csv")
win.describe()
win.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
win.data = win.ix[:,1:]
win.data.head(4)

# Normalizing the numerical data 
win_normal = scale(win.data)

pca = PCA(n_components = 13)
pca_values = pca.fit_transform(win_normal)


# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

# plot between PCA1 and PCA2 
x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:,2]
plt.scatter(x,y)
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.scatter(x,y,z)

################### Clustering  ##########################
new_df = pd.DataFrame(pca_values[:,0:14])

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_


