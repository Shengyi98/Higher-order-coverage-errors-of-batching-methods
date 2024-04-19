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

# generate waiting time process

def simtrue(lmbda,mu,C=10):
    waiting_times = np.zeros(C,)
    interarrival_times = np.random.exponential(1/lmbda,size = (C+2,))
    service_times = np.random.exponential(1/mu,size = (C+2,))
    last_time = 0
    for c in range(C):        
        last_time = max(last_time + service_times[c]-interarrival_times[c],0)
        waiting_times[c] = last_time
    return waiting_times

n_rep = 1000000
K = 5
delta = 0.1

lmbda = 1
mu = 2
level = 0.95
alpha = 1-level

N_list = [500,550,600,650,700,800,1000,1250,1500,1750,2000,2250,2500,3000,3500,4000,5000] # s2
coverages = np.zeros((len(N_list,)))
lengths = np.zeros((len(N_list),n_rep))

truth = np.average(simtrue(lmbda,mu,1000000)[1000:])
    
def trial(rep):
    covlen = np.zeros((1,len(N_list),2))
    
    for N_idx in range(len(N_list)):
        N = N_list[N_idx]
        X = simtrue(lmbda,mu,N+1000)[1000:]

        batch_gap = int((N/K)**delta)
        batch_data = int(N/K) - batch_gap
        batch_len = batch_data + batch_gap
        batch_means = np.zeros((K,))
        for k in range(K):
            start_idx = k*batch_len
            end_idx = k*batch_len + batch_data
            batch_means[k] = np.average(X[start_idx:end_idx])


        stat = np.sqrt(K)*(np.average(batch_means) - truth)/np.std(batch_means,ddof=1)
        if stat < t.ppf(1-alpha/2,K-1) and stat > t.ppf(alpha/2,K-1):
            covlen[0,N_idx,0] = 1
        if rep%10000 == 0:
            print(rep)
        covlen[0,N_idx,1] = np.std(batch_means,ddof=1)/np.sqrt(K)*t.ppf(alpha/2,K-1)
    return covlen
import multiprocessing
from multiprocessing import Pool
if __name__ == '__main__':
    with Pool(multiprocessing.cpu_count()) as p:

        results = p.map(trial, [rep for rep in range(n_rep)])

        result = np.concatenate(tuple(results))
        np.savez('MM195_1_2_s3',covlen = result)