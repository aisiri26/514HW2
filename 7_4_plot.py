from sklearn.cluster import k_means
import numpy as np 
import scipy as scipy
from scipy import linalg
import random
import matplotlib.pyplot as plt 


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

k = 3
A = np.random.permutation(A)
centroids = A[:k]

for x in range(1,10):
    y = k_means(A, n_clusters=x, max_iter=1000, return_n_iter=True)[2]
    #print(y)
    plt.scatter(x, y)

plt.show()
    
