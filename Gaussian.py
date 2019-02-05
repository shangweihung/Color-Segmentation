import numpy as np
import math



def Gaussian_likelihood(data, mean, cov, func_type):
    if func_type == 1:
   
         
         # perform cholesky decomposition
    
        decomp = np.linalg.cholesky(np.linalg.inv(cov))
        exponential = np.exp(-0.5 * np.sum (np.square (np.dot(data-mean,decomp)),axis = 1))
        likelihood=1/np.sqrt(((2*math.pi)**3) * np.linalg.det(cov)) * exponential
        
        return likelihood


    else:
        # for GMM 
        C,num_c=mean.shape
        #data=data.T

        likelihood=0
        for i in range(C):
            likelihood = likelihood + Gaussian_likelihood(data,mean[i,:],np.diagflat(cov[i,:]), 1)    

        return likelihood/C
