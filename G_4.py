import numpy as np
from scipy.stats import t
import math
from math import log
from math import exp
import math
import numpy as np
import random
from numpy.linalg import inv
from scipy.stats import t
from scipy.stats import norm
import itertools
import time
from scipy.stats import chi2
from scipy.stats import gamma
from random import choices
import pandas as pd
from sklearn.linear_model import LogisticRegression
import random

random.seed(10)

def psi(X,Y):
    allzero = True
    allone = True
    for yi in Y:
        if yi == 0:
            allone = False
        if yi == 1:
            allzero = False
    if allone:
        return 1
    if allzero:
        return 0
    clf = LogisticRegression(random_state=0).fit(X, Y)
    return clf.predict_proba(x)[0,1]


def simprocess(N,d):
    X = np.random.normal(size = (N,d))
    Y = np.zeros((N,))
    for i in range(N):
        p = 1/(1+exp(-np.average(X[i,:])))
        Y[i] = np.random.binomial(1,p)
    return [X,Y]

def trial(rep):
    
    covlen = np.zeros((1,len(N_list),2,4))
    for N_idx in range(len(N_list)):
        N = N_list[N_idx]

        X,Y = simprocess(N,d)

        batch_gap = 0
        batch_data = int(N/K) - batch_gap
        batch_len = batch_data + batch_gap
        batch_est = np.zeros((K,))
        for k in range(K):
            start_idx = k*batch_len
            end_idx = k*batch_len + batch_data
            batch_est[k] = psi(X[start_idx:end_idx],Y[start_idx:end_idx])
            
        emp_est = psi(X,Y)
        
        j_est = np.zeros((K,))
        for k in range(K):
            start_idx = k*batch_len
            end_idx = k*batch_len + batch_data
            data_X = np.concatenate((X[:start_idx],X[end_idx:]))
            data_Y = np.concatenate((Y[:start_idx],Y[end_idx:]))
            j_est[k] = K*emp_est - (K-1)*psi(data_X,data_Y)


        # batching
        stat = np.sqrt(K)*(np.average(batch_est) - truth)/np.std(batch_est,ddof=1)
        if stat < t.ppf(1-alpha/2,K-1) and stat > t.ppf(alpha/2,K-1):
            covlen[0,N_idx,0,0] = 1

        covlen[0,N_idx,1,0] = np.std(batch_est,ddof=1)/np.sqrt(K)*t.ppf(alpha/2,K-1)
        
        # sectioning
        
        
        stat = np.sqrt(K)*(emp_est - truth)/np.sqrt(np.average((batch_est-emp_est)**2)*K/(K-1))
        if stat < t.ppf(1-alpha/2,K-1) and stat > t.ppf(alpha/2,K-1):
            covlen[0,N_idx,0,1] = 1

        covlen[0,N_idx,1,1] = np.sqrt(np.average((batch_est-emp_est)**2)*K/(K-1))/np.sqrt(K)*t.ppf(alpha/2,K-1)
        
        # SB
        stat = np.sqrt(K)*(emp_est - truth)/np.std(batch_est,ddof=1)
        if stat < t.ppf(1-alpha/2,K-1) and stat > t.ppf(alpha/2,K-1):
            covlen[0,N_idx,0,2] = 1

        covlen[0,N_idx,1,2] = np.std(batch_est,ddof=1)/np.sqrt(K)*t.ppf(alpha/2,K-1)
        
        # SJ
        stat = np.sqrt(K)*(np.average(j_est) - truth)/np.std(j_est,ddof=1)
        if stat < t.ppf(1-alpha/2,K-1) and stat > t.ppf(alpha/2,K-1):
            covlen[0,N_idx,0,3] = 1
        
        covlen[0,N_idx,1,3] = np.std(j_est,ddof=1)/np.sqrt(K)*t.ppf(alpha/2,K-1)
    if rep%10000 == 0:
        print(rep)    
    return covlen

n_rep = 1000000
K = 10
delta = 0.25
d = 3
lmbda = 8
mu = 10
level = 0.8
alpha = 1-level

x = np.ones((1,d))


N_list = [60,70,80,90,100,125,150,175,200,250,300,350,400,500,600,700,800] # par2 K = 10


coverages = np.zeros((len(N_list,)))
lengths = np.zeros((len(N_list),n_rep))
truth = 1/(1+exp(-1))

import multiprocessing
from multiprocessing import Pool
if __name__ == '__main__':
    with Pool(multiprocessing.cpu_count()) as p:

        results = p.map(trial, [rep for rep in range(n_rep)])

        result = np.concatenate(tuple(results))
        np.savez('logistic_par2',covlen = result)
        
