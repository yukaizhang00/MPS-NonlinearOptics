# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:48:18 2021

@author: yukai

Second Trail of MPS
"""

import numpy as np
from numpy import linalg as LA

n = 4
c = np.arange(64.).reshape(2,2,2,2,2,2)
#c = np.array([[[1+1j,0],[0,2]],[[1,1],[1,0]],[[7,0],[0,2+1j]]])
c = c/np.sqrt(np.sum(np.power(c,2)))


#checking if two vectors are proportional
def isprop(a,b):
    p = []
    for i in range(len(a)):
        if a[i] == 0 and np.abs(b[i]) > 10e-12:
            return False
        if b[i] == 0 and np.abs(a[i]) > 10e-12:
            return False
        if np.abs(b[i]) > 10e-12 and np.abs(a[i]) > 10e-12:
            p.append(a[i]/b[i])
    m = np.mean(p)
    for i in p:
        if (i - m)/m > 1e-10:
            return False
    return True

#Proforming schmidt decomposition with SVD
def schvec(state, l):


def schdecomp(state):
    j = state.shape
    n = len(j)
    sumindex = [(i,) for i in range(j[1])]
    if n >= 2:
        for i in range(2,n):
            newindex = []
            for index in sumindex:
                for s in range(j[i]):    
                    newindex.append(index+(s,))
            sumindex = newindex
    spmatrix = np.zeros((j[0],len(sumindex)),dtype=np.complex_)
    for i in range(j[0]):
        for k in range(len(sumindex)):
            spmatrix[(i,k)] += state[(i,)+sumindex[k]]
    u, s, vh = LA.svd(spmatrix, full_matrices=False)
    '''
    print(spmatrix)
    print(u)
    print(s)
    print(vh)
    print('haha', np.dot(s,vh))
    print(np.dot(u, np.dot(np.diag(s),vh)))
    '''
    return u,s,vh, sumindex


u,s,vh, sumindex = schdecomp(c)
print(vh)
phi2 = []
for vec in vh:
    v = np.zeros((c.shape[1:]),dtype=np.complex_)
    for k in range(len(sumindex)):
        v[sumindex[k]] += vec[k]
    phi2.append(v)


for i in phi2:
    u,s,vh, inde = schdecomp(i)
    print('!!!!!!!!!!!!!!!!!!!!!!!!')
    print(i)
    print(np.dot(u, np.dot(np.diag(s),vh)))
    print('###############')
    print(vh)
    print(s)
'''
lambd = [s]
Gamma = [u]
for i in range(1,n):
    Gamma.append(numpy.array)
entry = [(v, 1)  for vec in vh]
while entry:
    last = entry[-1]
    
for alp in range(1,len(c.shape)):
    Gamma.append(numpy.array())
    for vec in range(len(vh)):
        u1,s1, vh1
        Gamma[-1] = 


schdecomp(c)
'''