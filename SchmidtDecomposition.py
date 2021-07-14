# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:48:18 2021

@author: yukai

First Trail of MPS
"""

import numpy as np
from numpy import linalg as LA

n = 4
c = np.arange(64.).reshape(2,2,2,2,2,2)
#c = np.array([[[1+1j,0],[0,0]],[[0,0],[0,2+1j]]])
c = c/np.sqrt(np.sum(np.power(c,2)))

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
    rou_u = np.zeros((j[0],j[0]),dtype=np.complex_)
    for i in range(j[0]):
        for k in range(j[0]):
            temp = 0
            for index in sumindex:
                temp += state[((i,)+index)] * np.conj(state[((k,)+index)])
            rou_u[(i,k)] = temp
    rou_v = np.zeros((len(sumindex),len(sumindex)),dtype=np.complex_)
    for i in range(len(sumindex)):
        for k in range(len(sumindex)):
            temp = 0
            for t in range(j[0]):
                temp += state[(t,) + sumindex[i]] * np.conj(state[(t,) + sumindex[k]])
            rou_v[(i,k)] = temp
    print(rou_v)
    print(rou_u)
    eigval_u, eigvec_u = LA.eig(rou_u)
    eigval_v, eigvec_v = LA.eig(rou_v)
    #Note you need to transpose eig here important
    eigvec_u = np.transpose(eigvec_u)
    eigvec_v = np.transpose(eigvec_v)
    print('Value',eigval_v)
    print('Vector',eigvec_v)
    #print('Scale',[np.sum(np.dot(rou_u,v))/np.sum(v) for v in eigval_v])
    print('Scale',[np.sum(np.dot(rou_v,v))/np.sum(v) for v in eigvec_v])    
    vec_u = [];vec_v = []
    lamb_u = [];lamb_v = []
    sortu = sorted(range(len(eigval_u)), key=lambda k: -np.abs(eigval_u[k]))
    for e in sortu:
        if np.abs(eigval_u[e]) <1e-5:
            break
        vec_u.append(eigvec_u[e]/(np.sqrt(np.sum(np.conj(eigvec_u[e])*eigvec_u[e]))))
        lamb_u.append(eigval_u[e])
    sortv = sorted(range(len(eigval_v)), key=lambda k: -np.abs(eigval_v[k]))
    for e in sortv:
        if np.abs(eigval_v[e]) <1e-5:
            break
        vec_v.append(eigvec_v[e]/(np.sqrt(np.sum(np.conj(eigvec_v[e])*eigvec_v[e]))))
        lamb_v.append(eigval_v[e])
    print(lamb_u, vec_u)
    print(lamb_v, vec_v)
    return lamb_u,lamb_v, vec_u,vec_v, sumindex

lamb_u,lamb_v, vec_u,vec_v, sumindex = schdecomp(c)
signs = [(+1.,),(-1.,)]
for i in range(len(lamb_u)-1):
    updatesigns = []
    for sign in signs:
        updatesigns.append(sign + (+1.,))
        updatesigns.append(sign + (-1.,))
    signs = updatesigns
posbres = []
for sign in signs:
    recreate = np.zeros(c.shape,dtype=np.complex_)
    for v in range(len(lamb_u)):
        for i in range(c.shape[0]):
            for k in range(len(sumindex)):
                recreate[(i,) + sumindex[k]] += sign[v]*np.sqrt(lamb_u[v])*vec_u[v][i]*vec_v[v][k]
    posbres.append(recreate)
dist = []
for res in posbres:
    diff = res - c
    dist.append(np.sum(np.conj(diff)*diff))
    if np.sum(np.conj(diff)*diff)<1e-5:
        print("Got it")