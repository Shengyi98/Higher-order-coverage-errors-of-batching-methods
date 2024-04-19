import numpy as np
from numpy.linalg import inv
import math
from scipy.stats import t
from scipy.stats import norm
import itertools
import time
from scipy.stats import chi2

dim = 2;


# provide gradients information
u = np.zeros((dim,));
v = np.zeros((dim,dim)); # 2*second derivative (i.e., 2*Hessian), should be symmetric
w = np.zeros((dim,dim,dim)); # 6*third derivative, should be symmetric

u = [1,2]
v[1,1] = 1
w[0,0,0] = 1

N = 1000
n_rep = 1000000
n_sim = 1000000
o_props = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
betas = [0.01,0.05,0.1,0.2,0.3,0.9,0.95]
beta_invs = [1/beta for beta in betas]

o_props_ext = [1]+o_props
n_rep2 = 100000
c_values = np.load('c_values.npy')

def target(x):
    return np.inner(u,x) + np.einsum('ij,i,j',v,x,x) + np.einsum('ijk,i,j,k',w,x,x,x)
ct = np.zeros((n_rep2,len(beta_invs),len(o_props_ext)))
for rep in range(n_rep2):      
    X1 = np.random.exponential(size=(N,)) - 1
    X2 = (chi2.rvs(df=1,size=(N,)) - 1)/math.sqrt(2)
    X = np.zeros((N,2))
    X[:,0] = X1
    X[:,1] = X2
    
    X_aves = np.average(X,axis=1);
    aves_all = np.average(X_aves,axis=0);
    
    # overlapping batching
    for beta_idx,beta_inv in enumerate(beta_invs):
        if beta_idx == 0:
            continue
        for prop_idx,o_prop in enumerate(o_props_ext):
            c_value = c_values[beta_idx,prop_idx]
            if c_value<0.01:
                continue
            beta = 1/beta_inv
            m = int(N * beta)
            d = int(max(1,m*(1-o_prop)))
            b = int((N-m)/d+1)
            ssq = 0
            batch_means = np.zeros((b,dim))
            batch_ests = np.zeros(b)
            emp_mean = np.average(X,axis=0)
            emp_est = target(emp_mean)
            for i in range(b):
                start_idx = i*d
                end_idx = start_idx + m
                batch_means[i] = np.average(X[start_idx:end_idx],axis = 0)
                batch_ests[i] = target(batch_means[i])
                ssq += (batch_ests[i]-emp_est)**2
            sigmaobi = ssq/(1-beta)*m/b
#             print(np.average(X[start_idx:end_idx],axis = 0))
#             print(batch_means[0])
            stat = np.sqrt(N)*(emp_est)/np.sqrt(sigmaobi)
            
#             if prop_idx == 0:
#                 c_value = c_values[beta_idx]
#             else:
#                 c_value = np.quantile(result_list[:,beta_idx,prop_idx],0.9)
            
            if stat>-c_value and stat<c_value:
                ct[rep,beta_idx,prop_idx] += 1
    if rep%10 == 0:
        print(rep)
np.save('cov',ct)