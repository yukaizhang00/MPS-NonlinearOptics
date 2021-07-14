`# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:48:18 2021

@author: yukai

Second Trail of MPS
"""

import numpy as np
from numpy import linalg as LA

n = 4
c = np.arange(64.).reshape(2,2,2,2,2,2)
c = np.array([[[1+1j,0],[0,2]],[[1,1],[1,0]],[[7,0],[0,2+1j]]])
c = np.random.rand(2,2,2,3,2,3)
d = np.random.rand(2,2,2,3,2,3) * 1j
c = c + d
c = c/np.sqrt(np.sum(np.conj(c) * c))


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

#Proforming schmidt decomposition with SVD for (H0*...*Hl)*(Hl+1*...*Hn)
def schdecomp(state, l):
    j = state.shape
    n = len(j)
    lsumindex = [(i,) for i in range(j[0])]
    rsumindex = [(i,) for i in range(j[l+1])]
    if l >= 1:
        for i in range(1,l+1):
            newindex = []
            for index in lsumindex:
                for s in range(j[i]):    
                    newindex.append(index+(s,))
            lsumindex = newindex
    if n >= l + 2:
        for i in range(l+2,n):
            newindex = []
            for index in rsumindex:
                for s in range(j[i]):    
                    newindex.append(index+(s,))
            rsumindex = newindex
    spmatrix = np.zeros((len(lsumindex),len(rsumindex)),dtype=np.complex_)
    for i in range(len(lsumindex)):
        for k in range(len(rsumindex)):
            spmatrix[(i,k)] += state[lsumindex[i]+rsumindex[k]]
    u, s, vh = LA.svd(spmatrix, full_matrices=False)
    '''
    print(spmatrix)
    print(u)
    print(s)
    print(vh)
    print('haha', np.dot(s,vh))
    '''
    #print(spmatrix)
    #print(np.dot(u, np.dot(np.diag(s),vh)))
    return u,s,vh, lsumindex, rsumindex



Gamma = []
S = []
N = len(c.shape)
#For the first Matrix
u,s,vh, lsumindex, rsumindex = schdecomp(c,0)
Gamma.append(u)
S.append(s)
cnext = np.zeros((len(s),) + c.shape[1:], dtype = np.complex_)
for i in range(len(s)):
    for j in range(len(rsumindex)):
        cnext[(i, ) + rsumindex[j]] += vh[(i,j)]
        
for i in range(1,N-1):
    u,s,vh,lsumindex, rsumindex = schdecomp(cnext, 1)
    S.append(s)
    #Note the index order for Gamma^[n] >= 2 are alpha_n-1, i_n, alpha_n
    uordered = np.zeros(cnext.shape[0:2]+ (len(s),), dtype=np.complex_)
    for j in range(len(lsumindex)):
        for k in range(len(s)):
            uordered[lsumindex[j] + (k, )] += u[j,k]
    Gamma.append(uordered)
    cnext = np.zeros((len(s),) + cnext.shape[2:], dtype = np.complex_)
    for k in range(len(s)):
        for j in range(len(rsumindex)):
            cnext[(k, ) + rsumindex[j]] += vh[(k,j)]
Gamma.append(cnext)
#print(Gamma)
recreat = Gamma[0]
for i in range(1, N):
    recreat = np.tensordot(recreat, np.diag(S[i-1]), (-1,0))
    recreat = np.tensordot(recreat, Gamma[i], (-1,0))
print([ss.shape for ss in Gamma])
print(np.sqrt(np.sum(np.conj(c) * c)))
print(np.sqrt(np.sum(np.conj(c-recreat) * (c-recreat))))
'''
print(vh)
phi2 = []
for vec in vh:
    v = np.zeros((c.shape[1:]),dtype=np.complex_)
    for k in range(len(rsumindex)):
        v[rsumindex[k]] += vec[k]
    phi2.append(v)


zz,xx,cc,vv,bb =schdecomp(c, 1)

for i in phi2:
    print(np.sqrt(np.sum(np.power(i,2))))
    u,s,vh, inde, g = schdecomp(i,0)
    print('!!!!!!!!!!!!!!!!!!!!!!!!')
    #print(i)
    #print(np.dot(u, np.dot(np.diag(s),vh)))
    print(np.dot(np.diag(s),vh))
    print('###############')
    
    print(len(vh))
    for veec in vh:
        for vec in cc:
            if isprop(vec,veec):
                print('Yah!!!!!!!!!!!!!!!!!!!!!!!')
    
    print(s)


#print(cc,s)

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