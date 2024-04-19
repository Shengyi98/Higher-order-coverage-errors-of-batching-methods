import numpy as np
from numpy.linalg import inv
import math
from scipy.stats import t
from scipy.stats import norm
import itertools
import time
from scipy.stats import chi2

print('Start')
dim = 1;


# provide gradients information
u = np.zeros((dim,));
v = np.zeros((dim,dim)); # 2*second derivative (i.e., 2*Hessian), should be symmetric
w = np.zeros((dim,dim,dim)); # 6*third derivative, should be symmetric

# u = np.ones((dim,));
# v = np.ones((dim,dim));
# w = np.ones((dim,dim,dim))


u[0] = 1
v[0,0] = 1

# provide moments and cumulants (suppose mean=0, otherwise reformulate the function)
sigma = np.zeros((dim,dim));
gamma = np.zeros((dim,dim,dim));
kappa = np.zeros((dim,dim,dim,dim));

sigma = np.identity(dim);


# sigma = np.matmul(np.array([[1/3,0,0],[1/3,1/2,0],[1/3,1/2,1]]),np.array([[1/3,0,0],[1/3,1/2,0],[1/3,1/2,1]]).T)
# gamma = ;
# kappa = ;

sigmapinv = inv(sigma[1:,1:])
sigmainv = inv(sigma)


# n_rep0 = 10000;
# n_rep = 10000;
max_K = 50
npz = np.load('results_normal.npz')
S_theos = npz['S']
B_theos = npz['B']
SB_theos = np.zeros_like(S_theos)
SJ_theos = npz['SJ']
Edgeworth_errors = npz['E']
S_actuals = npz['SA']
B_actuals = npz['BA']
SJ_actuals = npz['SJA']
SB_actuals = np.zeros_like(S_actuals)

for K in range(2,31):
    n = 50;
    level = 0.8;
    sym_level = 1-(1-level)/2;
    tsigma0 = (sigma[0,0] - np.einsum('ij,i,j',sigmapinv,sigma[0,1:],sigma[0,1:]) )/K
    q = t.ppf(sym_level,K-1);

    def edge_poly_1(x):
        return 1/6*np.einsum('ijk,il,jm,kn,l,m,n',gamma,sigmainv,sigmainv,sigmainv,x,x,x) \
               - 1/2*np.einsum('ij,ijk,kl,l',sigmainv,gamma,sigmainv,x)


    def edge_1(x):
        return 1/6*np.einsum('ijk,il,jm,kn,l,m,n',gamma,sigmainv,sigmainv,sigmainv,x,x,x) \
               - 1/2*np.einsum('ij,ijk,kl,l',sigmainv,gamma,sigmainv,x)

    def edge_2(x):
        return 1/24*np.einsum('ijkl,i,j,k,l',kappa,x,x,x,x) \
                -1/4*np.einsum('ijkk,i,j',kappa,x,x) \
                +1/8*np.einsum('iijj',kappa) \
                +1/72*np.einsum('ijk,lmn,i,j,k,l,m,n',gamma,gamma,x,x,x,x,x,x) \
                -6/72*np.einsum('ijk,lmm,i,j,k,l',gamma,gamma,x,x,x,x) \
                -9/72*np.einsum('ijk,lmk,i,j,l,m',gamma,gamma,x,x,x,x) \
                +9/72*np.einsum('ijj,kll,i,k',gamma,gamma,x,x) \
                +18/72*np.einsum('ijk,kll,i,j',gamma,gamma,x,x) \
                +18/72*np.einsum('ijk,ljk,i,l',gamma,gamma,x,x) \
                -9/72*np.einsum('iij,jkk',gamma,gamma) \
                -6/72*np.einsum('ijk,ijk',gamma,gamma)

    n_rep = 10000
    start = time.time()
    S_estimates = np.zeros((n_rep,))

    for rep in range(n_rep):
        # generate data (from its limiting normal distribution)
        X_aves_scaled = np.random.multivariate_normal(size=(K,),mean = np.zeros((dim,)),cov=sigma);
    #     X_aves_scaled = X/math.sqrt(n);
        A = np.average(X_aves_scaled,axis=0);
        B = np.zeros((K,dim));
        E2 = 0;
        for i in range(K):
            B[i,:] = X_aves_scaled[i,:] - A
            E2 = E2 + np.inner(u,B[i,:])**2
        qE2 = math.sqrt(E2)


        F_pos = q * qE2/math.sqrt(K*(K-1)) - np.inner(u[1:],A[1:])
        A[0] = F_pos
        a = np.inner(u,A)
        b1 = np.einsum('ij,i,j',v,A,A)
        b2 = np.einsum('ijk,i,j,k',w,A,A,A)
        b1p = 2*np.inner(v[0,:],A)
        ap = u[0];
        d = 0;
        e = 0;
        dp = 0;

        for i in range(K):

            ABi = A + B[i,:]
            sBi = np.einsum('ij,i,j',v,ABi,ABi);
            sBi2 = np.inner(u,B[i,:])
            d = d + 2*sBi2*sBi
            e = e + (sBi-b1)**2 + 2*sBi2*np.einsum('ijk,i,j,k',w,ABi,ABi,ABi);
            dp = dp + 4*sBi2*np.inner(v[0,:],B[i,:]);

        Fx = b1/qE2 -1/2*d*a/(qE2**3);
        Fxx = 1/qE2*(a*(-e/E2+3/4*(d**2)/(E2**2))-b1*d/E2+2*b2)
        Fy = ap/qE2
        Fxy = b1p/qE2 - 1/2*(dp*a+ap*d)/(qE2**3)

        yx = -Fx/Fy
        yxx = -(Fxx+2*Fxy*yx)/Fy

        p1a = 0
        for i in range(K):
            p1a = p1a + edge_poly_1(A+B[i,:])

        F_neg = -q * qE2/math.sqrt(K*(K-1)) - np.inner(u[1:],A[1:])
        A[0] = F_neg
        a = np.inner(u,A)
        b1 = np.einsum('ij,i,j',v,A,A)
        b2 = np.einsum('ijk,i,j,k',w,A,A,A)
        b1p = 2*np.inner(v[0,:],A)
        ap = u[0];
        d = 0;
        e = 0;
        dp = 0;

        for i in range(K):
            ABi = A + B[i,:]
            sBi = np.einsum('ij,i,j',v,ABi,ABi);
            sBi2 = np.inner(u,B[i,:])
            d = d + 2*sBi2*sBi
            e = e + (sBi-b1)**2 + 2*sBi2*np.einsum('ijk,i,j,k',w,ABi,ABi,ABi);
            dp = dp + 4*sBi2*np.inner(v[0,:],B[i,:]);


        Fx = b1/qE2 -1/2*d*a/(qE2**3);
        Fxx = 1/qE2*(a*(-e/E2+3/4*(d**2)/(E2**2))-b1*d/E2+2*b2)
        Fy = ap/qE2
        Fxy = b1p/qE2 - 1/2*(dp*a+ap*d)/(qE2**3)

        yx_neg = -Fx/Fy
        yxx_neg = -(Fxx+2*Fxy*yx_neg)/Fy

        p1a_neg = 0
        for i in range(K):
            p1a_neg = p1a_neg + edge_poly_1(A+B[i,:])

        mu = np.einsum('ij,i,j',sigmapinv,sigma[0,1:],A[1:])

        pdf0 = norm.pdf(F_pos-mu,scale = math.sqrt(tsigma0))
        pdf0_neg = norm.pdf(F_neg-mu,scale = math.sqrt(tsigma0))

        ER_1 = pdf0*p1a*yx - pdf0_neg*p1a_neg*yx_neg
        ER_2 = pdf0*(yxx/2) - pdf0_neg*(yxx_neg/2)
        ER_3 = 1/2*(pdf0*(-(F_pos-mu)/tsigma0)*(yx**2) - pdf0_neg*(-(F_neg-mu)/tsigma0)*(yx_neg**2))

        S_estimates[rep] = ER_1 + ER_2 + ER_3
    #     S_estimate1 = S_estimate1 + ER_1
    end = time.time()
    print(end-start)
    S_theos[K,0] = np.average(S_estimates)
    S_theos[K,1] = 1.96/math.sqrt(n_rep)*np.std(S_estimates)

    n_rep = 10000
    start = time.time()
    SJ_estimates = np.zeros((n_rep,))
    # SJ_estimate1 = 0
    for rep in range(n_rep):
        # generate data (from its limiting normal distribution)
        X_aves_scaled = np.random.multivariate_normal(size=(K,),mean = np.zeros((dim,)),cov=sigma);
    #     X_aves_scaled = X/math.sqrt(n);
        A = np.average(X_aves_scaled,axis=0);
        B = np.zeros((K,dim));
        E2 = 0;
        for i in range(K):
            B[i,:] = X_aves_scaled[i,:] - A
            E2 = E2 + np.inner(u,B[i,:])**2
        qE2 = math.sqrt(E2)


        F_pos = q * qE2/math.sqrt(K*(K-1)) - np.inner(u[1:],A[1:])
        A[0] = F_pos
        a = np.inner(u,A)
        uBs = np.zeros((K,))
        vBBs = np.zeros((K,));
        wABs = np.zeros((K,));
        vABs = np.zeros((K,))
        for i in range(K):
            uBs[i] = np.inner(u,B[i,:])
            vBBs[i] = np.einsum('ij,i,j',v,B[i,:],B[i,:])
            vABs[i] = np.einsum('ij,i,j',v,A-B[i,:]/(K-1),A-B[i,:]/(K-1))
            wABs[i] = np.einsum('ijk,i,j,k',w,A-B[i,:]/(K-1),A-B[i,:]/(K-1),A-B[i,:]/(K-1))
        b1 = np.einsum('ij,i,j',v,A,A) -np.sum(vBBs)/K/(K-1)
        b2 = K*np.einsum('ijk,i,j,k',w,A,A,A)-(K-1)/K*np.sum(wABs)
        b1p = 2*np.inner(v[0,:],A)
        ap = u[0];
        d = np.inner(uBs,vABs)*(-2)*(K-1);
        e = 0;
        dp = 0;

        for i in range(K):


            sBi2 = np.inner(u,B[i,:])

            e = e + ((K-1)**2)*(vABs[i] - np.average(vABs))**2 - 2*(K-1)*uBs[i]*wABs[i];
            dp = dp + 4*uBs[i]*np.inner(v[0,:],B[i,:]);

        Fx = b1/qE2 -1/2*d*a/(qE2**3);
        Fxx = 1/qE2*(a*(-e/E2+3/4*(d**2)/(E2**2))-b1*d/E2+2*b2)
        Fy = ap/qE2
        Fxy = b1p/qE2 - 1/2*(dp*a+ap*d)/(qE2**3)

        yx = -Fx/Fy
        yxx = -(Fxx+2*Fxy*yx)/Fy

        p1a = 0
        for i in range(K):
            p1a = p1a + edge_poly_1(A+B[i,:])

        #================

        F_neg = -q * qE2/math.sqrt(K*(K-1)) - np.inner(u[1:],A[1:])
        A[0] = F_neg
        a = np.inner(u,A)
        uBs = np.zeros((K,))
        vBBs = np.zeros((K,));
        wABs = np.zeros((K,));
        vABs = np.zeros((K,))
        for i in range(K):
            uBs[i] = np.inner(u,B[i,:])
            vBBs[i] = np.einsum('ij,i,j',v,B[i,:],B[i,:])
            vABs[i] = np.einsum('ij,i,j',v,A-B[i,:]/(K-1),A-B[i,:]/(K-1))
            wABs[i] = np.einsum('ijk,i,j,k',w,A-B[i,:]/(K-1),A-B[i,:]/(K-1),A-B[i,:]/(K-1))
        b1 = np.einsum('ij,i,j',v,A,A) -np.sum(vBBs)/K/(K-1)
        b2 = K*np.einsum('ijk,i,j,k',w,A,A,A)-(K-1)/K*np.sum(wABs)
        b1p = 2*np.inner(v[0,:],A)
        ap = u[0];
        d = np.inner(uBs,vABs)*(-2)*(K-1);
        e = 0;
        dp = 0;

        for i in range(K):


            sBi2 = np.inner(u,B[i,:])

            e = e + ((K-1)**2)*(vABs[i] - np.average(vABs))**2 - 2*(K-1)*uBs[i]*wABs[i];
            dp = dp + 4*uBs[i]*np.inner(v[0,:],B[i,:]);


        Fx = b1/qE2 -1/2*d*a/(qE2**3);
        Fxx = 1/qE2*(a*(-e/E2+3/4*(d**2)/(E2**2))-b1*d/E2+2*b2)
        Fy = ap/qE2
        Fxy = b1p/qE2 - 1/2*(dp*a+ap*d)/(qE2**3)

        yx_neg = -Fx/Fy
        yxx_neg = -(Fxx+2*Fxy*yx_neg)/Fy

        p1a_neg = 0
        for i in range(K):
            p1a_neg = p1a_neg + edge_poly_1(A+B[i,:])

        mu = np.einsum('ij,i,j',sigmapinv,sigma[0,1:],A[1:])

        pdf0 = norm.pdf(F_pos-mu,scale = math.sqrt(tsigma0))
        pdf0_neg = norm.pdf(F_neg-mu,scale = math.sqrt(tsigma0))

        ER_1 = pdf0*p1a*yx - pdf0_neg*p1a_neg*yx_neg
        ER_2 = pdf0*(yxx/2) - pdf0_neg*(yxx_neg/2)
        ER_3 = 1/2*(pdf0*(-(F_pos-mu)/tsigma0)*(yx**2) - pdf0_neg*(-(F_neg-mu)/tsigma0)*(yx_neg**2))

        SJ_estimates[rep] = ER_1 + ER_2 + ER_3
    #     SJ_estimate1 = SJ_estimate1 + ER_1
    end = time.time()
    print(end-start)
    SJ_theos[K,0] = np.average(SJ_estimates)
    SJ_theos[K,1] = 1.96/math.sqrt(n_rep)*np.std(SJ_estimates)

    n_rep = 10000
    start = time.time()
    B_estimates = np.zeros(n_rep)
    for rep in range(n_rep):
        # generate data (from its limiting normal distribution)
        X_aves_scaled = np.random.multivariate_normal(size=(K,),mean = np.zeros((dim,)),cov=sigma);
    #     X_aves_scaled = X/math.sqrt(n);
        A = np.average(X_aves_scaled,axis=0);
        B = np.zeros((K,dim));
        E2 = 0;
        for i in range(K):
            B[i,:] = X_aves_scaled[i,:] - A
            E2 = E2 + np.inner(u,B[i,:])**2
        qE2 = math.sqrt(E2)


        F_pos = q * qE2/math.sqrt(K*(K-1)) - np.inner(u[1:],A[1:])
        A[0] = F_pos
        a = np.inner(u,A)
        uBs = np.zeros((K,))
        wABs = np.zeros((K,))
        vABs = np.zeros((K,))
        for i in range(K):
            uBs[i] = np.inner(u,B[i,:])
            vABs[i] = np.einsum('ij,i,j',v,A+B[i,:],A+B[i,:])
            wABs[i] = np.einsum('ijk,i,j,k',w,A+B[i,:],A+B[i,:],A+B[i,:])
        b1 = np.average(vABs)
        b2 = np.average(wABs)
        b1p = 2*np.inner(v[0,:],A)
        ap = u[0]
        d = 2*np.inner(uBs,vABs)
        e = 0
        dp = 0

        for i in range(K):


            sBi2 = np.inner(u,B[i,:])

            e = e + (vABs[i] - np.average(vABs))**2 + 2*uBs[i]*wABs[i];
            dp = dp + 4*uBs[i]*np.inner(v[0,:],B[i,:]);

        Fx = b1/qE2 -1/2*d*a/(qE2**3);
        Fxx = 1/qE2*(a*(-e/E2+3/4*(d**2)/(E2**2))-b1*d/E2+2*b2)
        Fy = ap/qE2
        Fxy = b1p/qE2 - 1/2*(dp*a+ap*d)/(qE2**3)

        yx = -Fx/Fy
        yxx = -(Fxx+2*Fxy*yx)/Fy

        p1a = 0
        for i in range(K):
            p1a = p1a + edge_poly_1(A+B[i,:])

        #================

        F_neg = -q * qE2/math.sqrt(K*(K-1)) - np.inner(u[1:],A[1:])
        A[0] = F_neg
        a = np.inner(u,A)
        uBs = np.zeros((K,))
        wABs = np.zeros((K,))
        vABs = np.zeros((K,))
        for i in range(K):
            uBs[i] = np.inner(u,B[i,:])
            vABs[i] = np.einsum('ij,i,j',v,A+B[i,:],A+B[i,:])
            wABs[i] = np.einsum('ijk,i,j,k',w,A+B[i,:],A+B[i,:],A+B[i,:])
        b1 = np.average(vABs)
        b2 = np.average(wABs)
        b1p = 2*np.inner(v[0,:],A)
        ap = u[0]
        d = 2*np.inner(uBs,vABs)
        e = 0
        dp = 0

        for i in range(K):


            sBi2 = np.inner(u,B[i,:])

            e = e + (vABs[i] - np.average(vABs))**2 + 2*uBs[i]*wABs[i]
            dp = dp + 4*uBs[i]*np.inner(v[0,:],B[i,:])


        Fx = b1/qE2 -1/2*d*a/(qE2**3)
        Fxx = 1/qE2*(a*(-e/E2+3/4*(d**2)/(E2**2))-b1*d/E2+2*b2)
        Fy = ap/qE2
        Fxy = b1p/qE2 - 1/2*(dp*a+ap*d)/(qE2**3)

        yx_neg = -Fx/Fy
        yxx_neg = -(Fxx+2*Fxy*yx_neg)/Fy

        p1a_neg = 0
        for i in range(K):
            p1a_neg = p1a_neg + edge_poly_1(A+B[i,:])

        mu = np.einsum('ij,i,j',sigmapinv,sigma[0,1:],A[1:])

        pdf0 = norm.pdf(F_pos-mu,scale = math.sqrt(tsigma0))
        pdf0_neg = norm.pdf(F_neg-mu,scale = math.sqrt(tsigma0))

        ER_1 = pdf0*p1a*yx - pdf0_neg*p1a_neg*yx_neg
        ER_2 = pdf0*(yxx/2) - pdf0_neg*(yxx_neg/2)
        ER_3 = 1/2*(pdf0*(-(F_pos-mu)/tsigma0)*(yx**2) - pdf0_neg*(-(F_neg-mu)/tsigma0)*(yx_neg**2))

        B_estimates[rep] = ER_1 + ER_2 + ER_3
    #     B_estimate1 = B_estimate1 + ER_1
    end = time.time()
    print(end-start)
    B_theos[K,0] = np.average(B_estimates)
    B_theos[K,1] = 1.96/math.sqrt(n_rep)*np.std(B_estimates)
    
    n_rep = 10000
    start = time.time()
    SB_estimates = np.zeros((n_rep,))

    for rep in range(n_rep):
        # generate data (from its limiting normal distribution)
        X_aves_scaled = np.random.multivariate_normal(size=(K,),mean = np.zeros((dim,)),cov=sigma);
    #     X_aves_scaled = X/math.sqrt(n);
        A = np.average(X_aves_scaled,axis=0);
        B = np.zeros((K,dim));
        E2 = 0;
        for i in range(K):
            B[i,:] = X_aves_scaled[i,:] - A
            E2 = E2 + np.inner(u,B[i,:])**2
        qE2 = math.sqrt(E2)


        F_pos = q * qE2/math.sqrt(K*(K-1)) - np.inner(u[1:],A[1:])
        A[0] = F_pos
        uBs = np.zeros((K,))
        wABs = np.zeros((K,));
        vABs = np.zeros((K,))
        for i in range(K):
            uBs[i] = np.inner(u,B[i,:])
            vABs[i] = np.einsum('ij,i,j',v,A+B[i,:],A+B[i,:])
            wABs[i] = np.einsum('ijk,i,j,k',w,A+B[i,:],A+B[i,:],A+B[i,:])
        a = np.inner(u,A)
        b1 = np.einsum('ij,i,j',v,A,A)
        b2 = np.einsum('ijk,i,j,k',w,A,A,A)
        b1p = 2*np.inner(v[0,:],A)
        ap = u[0];
        d = 2*np.inner(uBs,vABs)
        e = 0;
        dp = 0;

        for i in range(K):

            ABi = A + B[i,:]
            sBi = np.einsum('ij,i,j',v,ABi,ABi);
            sBi2 = np.inner(u,B[i,:])
            e = e + (vABs[i] - np.average(vABs))**2 + 2*uBs[i]*wABs[i]
            dp = dp + 4*sBi2*np.inner(v[0,:],B[i,:]);

        Fx = b1/qE2 -1/2*d*a/(qE2**3);
        Fxx = 1/qE2*(a*(-e/E2+3/4*(d**2)/(E2**2))-b1*d/E2+2*b2)
        Fy = ap/qE2
        Fxy = b1p/qE2 - 1/2*(dp*a+ap*d)/(qE2**3)

        yx = -Fx/Fy
        yxx = -(Fxx+2*Fxy*yx)/Fy

        p1a = 0
        for i in range(K):
            p1a = p1a + edge_poly_1(A+B[i,:])

        F_neg = -q * qE2/math.sqrt(K*(K-1)) - np.inner(u[1:],A[1:])
        A[0] = F_neg
        uBs = np.zeros((K,))
        wABs = np.zeros((K,));
        vABs = np.zeros((K,))
        for i in range(K):
            uBs[i] = np.inner(u,B[i,:])
            vABs[i] = np.einsum('ij,i,j',v,A+B[i,:],A+B[i,:])
            wABs[i] = np.einsum('ijk,i,j,k',w,A+B[i,:],A+B[i,:],A+B[i,:])
        a = np.inner(u,A)
        b1 = np.einsum('ij,i,j',v,A,A)
        b2 = np.einsum('ijk,i,j,k',w,A,A,A)
        b1p = 2*np.inner(v[0,:],A)
        ap = u[0];
        d = 2*np.inner(uBs,vABs)
        e = 0;
        dp = 0;

        for i in range(K):
            ABi = A + B[i,:]
            sBi = np.einsum('ij,i,j',v,ABi,ABi);
            sBi2 = np.inner(u,B[i,:])
            
            e = e + (vABs[i] - np.average(vABs))**2 + 2*uBs[i]*wABs[i]
            dp = dp + 4*sBi2*np.inner(v[0,:],B[i,:]);


        Fx = b1/qE2 -1/2*d*a/(qE2**3);
        Fxx = 1/qE2*(a*(-e/E2+3/4*(d**2)/(E2**2))-b1*d/E2+2*b2)
        Fy = ap/qE2
        Fxy = b1p/qE2 - 1/2*(dp*a+ap*d)/(qE2**3)

        yx_neg = -Fx/Fy
        yxx_neg = -(Fxx+2*Fxy*yx_neg)/Fy

        p1a_neg = 0
        for i in range(K):
            p1a_neg = p1a_neg + edge_poly_1(A+B[i,:])

        mu = np.einsum('ij,i,j',sigmapinv,sigma[0,1:],A[1:])

        pdf0 = norm.pdf(F_pos-mu,scale = math.sqrt(tsigma0))
        pdf0_neg = norm.pdf(F_neg-mu,scale = math.sqrt(tsigma0))

        ER_1 = pdf0*p1a*yx - pdf0_neg*p1a_neg*yx_neg
        ER_2 = pdf0*(yxx/2) - pdf0_neg*(yxx_neg/2)
        ER_3 = 1/2*(pdf0*(-(F_pos-mu)/tsigma0)*(yx**2) - pdf0_neg*(-(F_neg-mu)/tsigma0)*(yx_neg**2))

        SB_estimates[rep] = ER_1 + ER_2 + ER_3
    #     S_estimate1 = S_estimate1 + ER_1
    end = time.time()
    print(end-start)
    SB_theos[K,0] = np.average(SB_estimates)
    SB_theos[K,1] = 1.96/math.sqrt(n_rep)*np.std(SB_estimates)
    

    Edgeworth_errors[K,0] = 0
    Edgeworth_errors[K,1] = 0

    n_rep2 = 1000000 # suggest 1,000,000
    ct_S = 0
    ct_SJ = 0
    ct_B = 0
    ct_SB = 0
    # estimate actual coverage
    def target(x):
        return np.inner(u,x) + np.einsum('ij,i,j',v,x,x) + np.einsum('ijk,i,j,k',w,x,x,x)
#     def target(x):
#         return math.sin(x[0] + x[1]**2)
    for rep in range(n_rep2):
        # provide distribution; generate data
    #     X = np.random.multivariate_normal(size=(K,n),mean = np.zeros((dim,)),cov=sigma);
        X_aves = np.random.normal(size=(K,1))/np.sqrt(n)
        aves_all = np.average(X_aves,axis=0)

        # Sectioning
        var_sec = 0
        pnt_est = target(aves_all)

        sec_vals = np.zeros((K,))
        for i in range(K):
            sec_vals[i] = target(X_aves[i,:])
            var_sec = var_sec + (sec_vals[i]-pnt_est)**2

        obj_val = math.sqrt(K*(K-1))*pnt_est/math.sqrt(var_sec)



        if (-q<obj_val)&(obj_val<q):
            ct_S = ct_S + 1

        # Batching
        pnt_B = np.average(sec_vals)
        var_B = 0
        for i in range(K):
            var_B = var_B + (sec_vals[i]-pnt_B)**2
        obj_B = math.sqrt(K*(K-1))*pnt_B/math.sqrt(var_B)

        if (-q<obj_B)&(obj_B<q):
            ct_B = ct_B + 1

        # Sectioned Jackknife
        Js = np.zeros((K,))
        for i in range(K):
            Js[i] = K*pnt_est - (K-1)*target((K*aves_all - X_aves[i,:])/(K-1))
        pnt_SJ = np.average(Js)
        var_SJ = 0
        for i in range(K):
            var_SJ = var_SJ + (Js[i]-pnt_SJ)**2
        obj_SJ = math.sqrt(K*(K-1))*pnt_SJ/math.sqrt(var_SJ)

        if (-q<obj_SJ)&(obj_SJ<q):
            ct_SJ = ct_SJ + 1
            
        obj_SB = math.sqrt(K*(K-1))*pnt_est/math.sqrt(var_B)

        if (-q<obj_SB)&(obj_SB<q):
            ct_SB = ct_SB + 1
        
        if rep%100000 == 0 :
            print(rep)

    S_actuals[K] = ct_S/n_rep2
    B_actuals[K] = ct_B/n_rep2
    SJ_actuals[K] = ct_SJ/n_rep2
    SB_actuals[K] = ct_SB/n_rep2
    
    print('Batching: '+ str(B_actuals[K])+ ' Sectioning '+str(S_actuals[K])+' Sectioned Jackknife ' +str(SJ_actuals[K])+' SB ' +str(SB_actuals[K]))
    print('BatchingT: '+ str(level+1/n*(Edgeworth_errors[K,0]+B_theos[K,0]))+ ' SectioningT '+str(level+1/n*(Edgeworth_errors[K,0]+S_theos[K,0]))+' Sectioned JackknifeT ' +str(level+1/n*(Edgeworth_errors[K,0]+SJ_theos[K,0]))+' SB ' +str(level+1/n*(Edgeworth_errors[K,0]+SB_theos[K,0])))

np.savez('results_normal80',S=S_theos,B=B_theos,SB=SB_theos,SJ=SJ_theos,E=Edgeworth_errors,SA =S_actuals, BA = B_actuals, SJA = SJ_actuals,SBA = SB_actuals)
