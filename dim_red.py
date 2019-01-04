import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model

data = pandas.read_csv('/Users/snehamitta/Desktop/ML/Assignment4/ChicagoDiabetes.csv', delimiter = ',')

X = data[['Crude Rate 2000','Crude Rate 2001','Crude Rate 2002','Crude Rate 2003',
	      'Crude Rate 2004','Crude Rate 2005','Crude Rate 2006','Crude Rate 2007',
	      'Crude Rate 2008','Crude Rate 2009','Crude Rate 2010','Crude Rate 2011']]

#Q1.a) To find the no. of observations and variables

nObs = X.shape[0]
nVar = X.shape[1]

print('The number of Observations are', nObs)
print('The number of Variables are', nVar)

#Q1.b) To generate the scatterplot of the variables

pandas.plotting.scatter_matrix(X, figsize=(5,5), c = 'red', diagonal = 'hist', hist_kwds = {'color':['burlywood']})
plt.show()

#Q1.c) To plot the Explained Variances against their indices  

XCorrelation = X.corr(method = 'pearson', min_periods = 1)

print('Empirical Correlation: \n', XCorrelation)

# Extract the Principal Components
_thisPCA = decomposition.PCA(n_components = nVar)
_thisPCA.fit(X)

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)

print('Explained Variance: \n', _thisPCA.explained_variance_)
print('Explained Variance Ratio: \n', _thisPCA.explained_variance_ratio_)
print('Cumulative Explained Variance Ratio: \n', cumsum_variance_ratio)
print('Principal Components: \n', _thisPCA.components_)

plt.plot(_thisPCA.explained_variance_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

plt.plot(_thisPCA.explained_variance_ratio_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

#Q1.d) To plot the Cumulative Sum of the Explained Variances against their indices

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)
x = range(1,13)
plt.plot(x,cumsum_variance_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.grid(True)
plt.show()

#Q1.e) Percentage of total variance explained by the first two Principal Components
print ('Percantage is :', 
	(_thisPCA.explained_variance_[0]+_thisPCA.explained_variance_[1])/sum(_thisPCA.explained_variance_))


#Q1.f) Elbow and silhouette values

first2PC = _thisPCA.components_[:, [0,1]]
print('Principal COmponent: \n', first2PC)

# Transform the data using the first two principal components
_thisPCA = decomposition.PCA(n_components = 2)
X_transformed = pandas.DataFrame(_thisPCA.fit_transform(X))

# Find clusters from the transformed data
maxNClusters = 10

nClusters = numpy.zeros(maxNClusters-1)
Elbow = numpy.zeros(maxNClusters-1)
Silhouette = numpy.zeros(maxNClusters-1)
TotalWCSS = numpy.zeros(maxNClusters-1)
Inertia = numpy.zeros(maxNClusters-1)

for c in range(maxNClusters-1):
   KClusters = c + 2
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20181010).fit(X_transformed)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(X_transformed, kmeans.labels_)
   else:
       Silhouette[c] = float('nan')

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = X_transformed.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])
      TotalWCSS[c] += WCSS[k]

   print("The", KClusters, "Cluster Solution Done")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(maxNClusters-1):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

# Draw the Elbow and the Silhouette charts  
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

kmeans = cluster.KMeans(n_clusters=4, random_state=20181010).fit(X_transformed)
X_transformed['Cluster ID'] = kmeans.labels_
print('The number of communities in each cluster are:', X_transformed.groupby('Cluster ID').count())
print('Names of communities in cluster 0:', data[X_transformed['Cluster ID']==0]['Community'])
print('Names of communities in cluster 1:', data[X_transformed['Cluster ID']==1]['Community'])
print('Names of communities in cluster 2:', data[X_transformed['Cluster ID']==2]['Community'])
print('Names of communities in cluster 3:', data[X_transformed['Cluster ID']==3]['Community'])

# Draw the first two PC using cluster label as the marker color 
carray = ['red', 'orange', 'green', 'black']
year = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']
rate = [25.4, 25.8, 27.2, 25.4, 26.2, 26.6, 27.4, 28.7, 27.9, 27.5, 26.8, 25.6]

for i in range(4):
    subData = X[X_transformed['Cluster ID'] == i]
    means=[]
    for j in range(subData.shape[1]):
        means.append(subData.iloc[:,j].mean())
    
    plt.plot(year, means, marker='o', linewidth=2, label="Cluster"+str(i))

plt.xlabel('Year')
plt.ylabel('crude hospitalization rate')
plt.plot(year, rate, marker='x', linewidth=2, label = "annual")     
plt.legend()   
plt.grid(True)
plt.show()











