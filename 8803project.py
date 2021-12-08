#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:36:50 2021

@author: ryx
"""

import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm
from random import choices
from numpy.linalg import inv

np.random.seed(1431)
T = 3000
K = 10 # K assets
d = 5 # d industry parameters
Alpha = [1/3,2/3,1] # risk averse parameter

# setting asset parameters
true_mu = [np.matrix(np.random.exponential(1, d)).reshape(-1,1) for k in range(K)]
true_sigma = [np.random.uniform(0,0.1) for k in range(K)]

# setting future context and risk perference
X = [np.matrix(np.random.uniform(-1,1,d)).reshape(-1,1) for t in range(T)]
alpha = choices(Alpha, k=T)

def posterior_update(t,k,r):
    C_i_k_old = C_i[k].copy()
    m_i_k_old = m_i[k].copy()
    # update
    T_i[k] += 1
    C_i[k] += X[t] * X[t].T
    m_i[k] = inv(C_i[k]) * (C_i_k_old * m_i_k_old + X[t] * r)
    A_i[k] += 1/2
    B_i[k] += 1/2 * (m_i_k_old.T * C_i_k_old * m_i_k_old - m_i[k].T * C_i[k] * m_i[k] + r**2)[0,0]

# Thompson sampling
# Initialization
T_i = [0] * K
C_i =[np.matrix(np.identity(d)) for i in range(K)]
m_i = [np.matrix(np.zeros((d,1))).reshape(-1,1) for i in range(K)] 
A_i = [1/2] * K
B_i = [1/2] * K
R = 0
Rs = []
for t in range(T):
    a_opt = -1
    CVaR_opt = -float('inf')
    true_a_opt = -1
    true_CVaR_opt = -float('inf')
    for i in range(K):
        lam_i = np.random.gamma(A_i[i], B_i[i]) 
        mu_i = np.matrix(np.random.multivariate_normal(np.array(m_i[i]).reshape(-1), 1/lam_i * inv(C_i[i]))).reshape(-1,1)
        CVaR_i = (alpha[t] * mu_i.T * X[t])[0,0] -  (2 * np.pi * lam_i)**(-0.5) * np.e**(-0.5 * norm.ppf(alpha[t])**2)
        if CVaR_i >= CVaR_opt:  
            CVaR_opt = CVaR_i
            a_opt = i
        true_CVaR_i = (alpha[t] * true_mu[i].T * X[t])[0,0] -  true_sigma[i] * (2 * np.pi)**(-0.5) * np.e**(-0.5 * norm.ppf(alpha[t])**2)
        if true_CVaR_i >= true_CVaR_opt:  
            true_CVaR_opt = true_CVaR_i
            true_a_opt = i
    # play arm a_opt
    r = np.random.normal(X[t].T * true_mu[a_opt], true_sigma[a_opt])
    # update posterior
    posterior_update(t,a_opt,r)
    # regret
    R += true_CVaR_opt - (alpha[t] * true_mu[a_opt].T * X[t])[0,0] +  true_sigma[a_opt] * (2 * np.pi)**(-0.5) * np.e**(-0.5 * norm.ppf(alpha[t])**2)
    Rs += [R]
            
plt.plot(range(1,T+1),Rs)    
plt.xlabel('Time')
plt.ylabel('Regret')
plt.title('Cumulative Risk-averse Regret')
plt.savefig('Regret.png', dpi=300)
plt.show() 


# # Random sampling
# # Initialization
# T_i = [0] * K
# C_i =[np.matrix(np.identity(d)) for i in range(K)]
# m_i = [np.matrix(np.zeros((d,1))).reshape(-1,1) for i in range(K)] 
# A_i = [1/2] * K
# B_i = [1/2] * K
# R = 0
# Rs_random = []
# for t in range(T):
#     a_opt = np.random.randint(K, size=1)[0]
#     true_a_opt = -1
#     true_CVaR_opt = -float('inf')
#     for i in range(K):
#         true_CVaR_i = (alpha[t] * true_mu[i].T * X[t])[0,0] -  true_sigma[i] * (2 * np.pi)**(-0.5) * np.e**(-0.5 * norm.ppf(alpha[t])**2)
#         if true_CVaR_i >= true_CVaR_opt:  
#             true_CVaR_opt = true_CVaR_i
#             true_a_opt = i
#     # play arm a_opt
#     r = np.random.normal(X[t].T * true_mu[a_opt], true_sigma[a_opt])
#     # update posterior
#     posterior_update(t,a_opt,r)
#     # regret
#     R += true_CVaR_opt - (alpha[t] * true_mu[a_opt].T * X[t])[0,0] +  true_sigma[a_opt] * (2 * np.pi)**(-0.5) * np.e**(-0.5 * norm.ppf(alpha[t])**2)
#     Rs_random += [R]

# plt.plot(range(1,T+1),Rs,label='Thompson Sampling')  
# plt.plot(range(1,T+1),Rs_random,label='Random Sampling')     
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Regret')
# plt.title('Cumulative Risk-averse Regret')
# plt.savefig('RegretCompare.png', dpi=300)
# plt.show() 

# # Save Data
# import pandas as pd

# df_mu = pd.DataFrame([np.array(mu).reshape(-1).tolist() for mu in true_mu], index=['Asset_'+str(j) for j in range(1,K+1)], columns = ['mu_'+str(i) for i in range(1,d+1)])
# df_sigma = pd.DataFrame(true_sigma, index=['Asset_'+str(j) for j in range(1,K+1)], columns = ['sigma'])

# df =  pd.concat([df_mu, df_sigma], axis=1)
# df.round(2).to_csv('parameters.csv', index=True)

# df_context = pd.DataFrame([np.array(x).reshape(-1).tolist() for x in X], index=['time '+str(t) for t in range(1,T+1)], columns = ['X_'+str(i) for i in range(1,d+1)])
# df_alpha = pd.DataFrame(alpha, index=['time '+str(t) for t in range(1,T+1)], columns = ['alpha'])

# df =  pd.concat([df_context, df_alpha], axis=1)
# df.round(2).to_csv('ContextAndAlpha.csv', index=True)