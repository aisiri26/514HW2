import numpy as np 
import scipy as scipy
from scipy import linalg
import random

x = np.full((50,50),0.7)
y = np.full((50,50),0.7)
z = np.full((50,50),0.7)
A = scipy.linalg.block_diag(x,y,z)
A[A ==0] = 0.3
B = np.random.rand(150,150)

for x in range(0,150):
    for y in range(0,150):
        if A[x,y]< B[x,y]:
            A[x,y]=1
        else:
            A[x,y]=0

A = np.random.permutation(A)
centroids = A[:k]

def hasConverged(upd, prev):
    if(np.array_equal(upd,prev)):
        converged=True
    else:
        converged=False
    return converged
    
def kmeans(matrix, k, c):
    initCentroids = centroids
    prevCentroids = np.zeros(initCentroids.shape)
    updatedCentroids=c
    clusters = np.zeros(150)
    dist = np.zeros((150,k))
    while hasConverged(updatedCentroids, prevCentroids)==False:
        for x in range(0,k):
            dist[:,x] = np.linalg.norm(matrix - initCentroids[x], axis=1)
        clusters = np.argmin(dist, axis = 1)
        i=0
        for x in updatedCentroids:  
            prevCentroids[i]=x
            i=i+1
        for x in range(0,k):
            updatedCentroids[x] = np.mean(matrix[clusters == x], axis=0)
    return updatedCentroids

print(kmeans(A, 3, centroids))
