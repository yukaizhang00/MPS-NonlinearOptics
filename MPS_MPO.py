# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:48:18 2021

@author: yukai

Second Trail of MPS
"""

import numpy as np
import math
from numpy import linalg as LA
from scipy import linalg as SLA

cutoff = 1e-4

n = 4
#c = np.arange(64.).reshape(2,2,2,2,2,2)
#c = np.array([[[1+1j,0],[0,2]],[[1,1],[1,0]],[[7,0],[0,2+1j]]])
sshape = (2,2,6,3,2,3,5)
sshape = (2,2,2,2,2,2,2,2,2,2)
c = np.random.rand(*sshape) + np.random.rand(*sshape) * 1j
c = c/np.sqrt(np.sum(np.conj(c) * c))


#Generate a random hermitian operator o_123...1'2'3'...
index = [(i,) for i in range(sshape[0])]
for j in range(1,len(sshape)):
    newind = []
    for k in index:
        for t in range(sshape[j]):
            newind.append(k+(t, ))
    index = newind
mm = np.random.rand(len(index),len(index))+ np.random.rand(len(index),len(index))*1j
mm += np.transpose(np.conj(mm))
o = np.zeros(sshape+ sshape,dtype=np.complex_)
for i in range(len(index)):
    for j in range(len(index)):
        o[index[i] + index[j]] += mm[i][j]

def generate_index(shape):
    l = len(shape)
    index = [(i,) for i in range(shape[0])]
    for i in range(1,l):
        newindex = []
        for ind in index:
            for s in range(shape[i]):    
                newindex.append(ind+(s,))
        index = newindex
    return index
    
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

#Preforming schmidt decomposition with SVD for (H_0 x ... x H_l)*(H_l+1 x ... x H_n)
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
    deleteindex = []
    for lamb in range(len(s)):
        if np.conj(s[lamb])*s[lamb] < cutoff * np.sqrt(np.sum(np.conj(s) * s)):
            deleteindex.append(lamb)
    u = np.delete(u,deleteindex, 1)
    s = np.delete(s,deleteindex, 0)
    vh = np.delete(vh,deleteindex, 0)
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

#Attempt for MPS
def MPS(c):
    Gamma = []
    S = []
    N = len(c.shape)
    #For the first Matrix
    u,s,vh, lsumindex, rsumindex = schdecomp(c,0)
    #Nomalize the S into U Notice U might not be consist of unit vectors
    #u = u * np.sqrt(np.sum(np.conj(s)*s))
    #s= s/np.sqrt(np.sum(np.conj(s)*s))
    for i in range(len(s)):
        coef = np.sqrt(np.sum(np.conj(u[...,i])*u[...,i]))
        s[i] = s[i]*coef.real
        u[...,i] = u[...,i]/coef
    Gamma.append(u)
    S.append(s)
    cnext = np.zeros((len(s),) + c.shape[1:], dtype = np.complex_)
    for i in range(len(s)):
        for j in range(len(rsumindex)):
            cnext[(i, ) + rsumindex[j]] += vh[(i,j)]
    left = np.tensordot(Gamma[0], np.diag(S[0]), (-1,0))
    for i in range(1,N-1):
        u,s,vh,lsumindex, rsumindex = schdecomp(cnext, 1)
        #Nomalize the S into U Notice U might not be consist of unit vectors
        #print(left.shape, 'bbbb', (len(S[-1]), u.shape[0]/len(S[-1]),len(s)))
        lleft = np.tensordot(left, u.reshape(len(S[-1]), u.shape[0]//len(S[-1]),len(s)), (-1,0))
        for k in range(len(s)):
            #print(s.shape, u.shape)
            #ut = np.tensordot(np.diag(S[-1]),u.reshape(len(S[-1]), u.shape[0]//len(S[-1]),len(s)),(-1,0)).reshape(u.shape)
            #coef = np.sqrt(np.sum(np.conj(ut[...,k])*ut[...,k]))
            coef = np.sqrt(np.sum(np.conj(lleft[...,k])*lleft[...,k]))
            s[k] = s[k]*coef.real
            u[...,k] = u[...,k]/coef
            #print('i is ',i,'with coef', coef)
        S.append(s)
        #Note the index order for Gamma^[n] >= 2 are alpha_n-1, i_n, alpha_n
        uordered = np.zeros(cnext.shape[0:2]+ (len(s),), dtype=np.complex_)
        for j in range(len(lsumindex)):
            for k in range(len(s)):
                uordered[lsumindex[j] + (k, )] += u[j,k]
        left = np.tensordot(np.tensordot(left, uordered, (-1, 0)), np.diag(s), (-1, 0))
        Gamma.append(uordered)
        cnext = np.zeros((len(s),) + cnext.shape[2:], dtype = np.complex_)
        for k in range(len(s)):
            for j in range(len(rsumindex)):
                cnext[(k, ) + rsumindex[j]] += vh[(k,j)]
    Gamma.append(cnext)
    #print(Gamma)
    #The following comment part is to check how well the decomposition is.
    '''
    recreat = Gamma[0]
    for i in range(1, N):
        recreat = np.tensordot(recreat, np.diag(S[i-1]), (-1,0))
        recreat = np.tensordot(recreat, Gamma[i], (-1,0))
    print('The original c norm',np.sqrt(np.sum(np.conj(c) * c)))
    print('c - MPS norm',np.sqrt(np.sum(np.conj(c-recreat) * (c-recreat))))
    '''
    print([ss.shape for ss in Gamma])
    return Gamma, S






#Attempt for MPO
#First step swap the index 1,2,3,...1',2',3' to 1,1',2,2',3,3' (o_mpo)
def swapindex(o):
    halfshape = o.shape[:int(len(o.shape)/2)]
    newshape = tuple()
    for i in halfshape:
        newshape += (i,i,)
    o_mpo = np.zeros(newshape, dtype=np.complex_)
    for i in index:
        for j in index:
            ind = tuple()
            for ii in range(len(i)):
                ind += (i[ii],)
                ind += (j[ii],)
            o_mpo[ind] += o[i+j]
    return o_mpo


def MPO(o):
    o_mpo = swapindex(o)
    combinds = []    
    #Second step review (1,1'),(2,2'),(3,3') as new index (o_sec):
    for i in o_mpo.shape[::2]:
        tempind = []
        for ii in range(i):
            tempind += [(ii, j) for j in range(i)]
        combinds.append(tempind)
    o_sec = np.zeros([len(ind) for ind in combinds], dtype=np.complex_)
    sumindex = generate_index(o_sec.shape)
    for ind in sumindex:
        corind = tuple()
        for n in range(len(ind)):
            corind += combinds[n][ind[n]]
        o_sec[ind] += o_mpo[corind]
    #Now similar as MPS we use SVD to get the MPO except we absorb S into the V:
    O = []
    N = len(o_sec.shape)
    #For the first Matrix
    u,s,vh, lsumindex, rsumindex = schdecomp(o_sec,0)
    vh = np.dot(np.diag(s), vh)
    #Reshape U back to 1,1' from (1,1')
    ures = np.zeros((int(np.sqrt(u.shape[0])) , int(np.sqrt(u.shape[0]))) + (len(s),), dtype = np.complex_)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            ures[combinds[0][i] + (j,)] += u[(i,j)]
    
    O.append(ures)
    cnext = np.zeros((len(s),) + o_sec.shape[1:], dtype = np.complex_)
    for i in range(len(s)):
        for j in range(len(rsumindex)):
            cnext[(i, ) + rsumindex[j]] += vh[(i,j)]
            
    for i in range(1,N-1):
        u,s,vh,lsumindex, rsumindex = schdecomp(cnext, 1)
        vh = np.dot(np.diag(s), vh)
        #Note the index order for Gamma^[n] >= 2 are alpha_n-1, i_n, i'_n,  alpha_n
        uordered = np.zeros((cnext.shape[0], ) + (int(np.sqrt(cnext.shape[1])) ,int(np.sqrt(cnext.shape[1])))+ (len(s),), dtype=np.complex_)
        for j in range(len(lsumindex)):
            for k in range(len(s)):
                uordered[(lsumindex[j][0], ) + combinds[i][lsumindex[j][1]] + (k, )] += u[j,k]
        O.append(uordered)
        cnext = np.zeros((len(s),) + cnext.shape[2:], dtype = np.complex_)
        for k in range(len(s)):
            for j in range(len(rsumindex)):
                cnext[(k, ) + rsumindex[j]] += vh[(k,j)]
    
    cres = np.zeros( (cnext.shape[0],)+(int(np.sqrt(cnext.shape[1])) , int(np.sqrt(cnext.shape[1]))), dtype = np.complex_)
    for i in range(cnext.shape[0]):
        for j in range(cnext.shape[1]):
            cres[(i,) + combinds[-1][j]] += cnext[(i,j)]
    O.append(cres)
    #print(Gamma)
    
    recrO = O[0]
    for i in range(1, N):
        recrO = np.tensordot(recrO, O[i], (-1,0))
    print([ss.shape for ss in O])
    print('The original O norm',np.sqrt(np.sum(np.conj(o_mpo) * o_mpo)))
    print('MPO norm',np.sqrt(np.sum(np.conj(recrO) * recrO)))
    print('O - MPO norm',np.sqrt(np.sum(np.conj(o_mpo-recrO) * (o_mpo-recrO))))
    return O


#Attempt to update MPS after One-mode operator
'''
G,SS = MPS(c)
l = len(G)-2
'''
#Well this is the easy part since it is just a simple tensor multiplication, gonna do it later. l_(> o <)_l
def one_mode(G, SS, V, l):
    if l != 0 and l != len(G) - 1:
        G[l] = np.einsum('ij,aib->ajb',V, G[l])
    elif l == 0:
        G[l] = np.einsum('ij,ib->jb',V, G[l])
    elif l == len(G) - 1:
        G[l] = np.einsum('ij,ai->aj',V, G[l])
    return G, SS


#Attempt to update MPS after Two-mode operator
#Generation of random two-mode hermitian operator at l: V_[i,j,i',j'] where i,j is the old state
'''
temp =np.random.rand(c.shape[l]* c.shape[l+1], c.shape[l] * c.shape[l+1]) + np.random.rand(c.shape[l]* c.shape[l+1], c.shape[l] * c.shape[l+1])*1.j
temp = temp + np.transpose(np.conj(temp))
V = temp.reshape((c.shape[l], c.shape[l+1], c.shape[l], c.shape[l+1]))
'''


def two_mode(G, SS, V, l, err = 1e-16, normalize = False):
    #The function input G is Gamma tensors and SS is Lambdas
    #V is two mode operator, l is location of V at l and l+1 (0 <= l <= N-2)
    #The function will return new Gammas and Lambdas
    Gl = G[l]
    #Here SS[l+1] or SS{l]
    Gm = np.tensordot(np.diag(SS[l]), G[l+1], (0,0))
    Tbefore = np.tensordot(Gl,Gm,(-1,0))
    #print('The original norm is', np.sqrt(np.sum(np.conj(Tbefore)*Tbefore)))
    if l != 0 and l != len(G)-2:
        #Calculate Theta^ij_alphagamma with natural index order [alpha, i, j, gamma]:
        Theta = np.einsum('ijklm,jka->ilma',np.tensordot(Gl, V,(1,0)),Gm)
        Tshape = Theta.shape
        #Mtheta is matrix form with index (alpha, i) x (j, gamma)
        Mtheta = Theta.reshape((Tshape[0]*Tshape[1], Tshape[2]*Tshape[3]))
        u,s,v = LA.svd(Mtheta, full_matrices=False)
        tots = np.sqrt(np.sum(np.power(s, 2)))
        '''
        for cut in range(len(s)):
            if np.sqrt(np.sum(np.power(s[:cut],2)))/tots > 1 - err:
                break
        print('norm of s is ', tots)
        s = s[:cut]* tots /np.sqrt(np.sum(np.power(s[:cut],2)))
        u = np.transpose(np.transpose(u)[:cut])
        v = v[:cut]
        '''
        if len(s) > 40:
            s = s[:40] * tots /np.sqrt(np.sum(np.power(s[:40],2)))
            u = np.transpose(np.transpose(u)[:40])
            v = v[:40]
        Gl_new = u.reshape(Tshape[0], Tshape[1], len(s))
        Gm_new = v.reshape(len(s), Tshape[2], Tshape[3])
        recrTheta = np.tensordot(np.tensordot(Gl_new, np.diag(s),(2,0)), Gm_new, (2,0))
    elif l == 0:
        Theta = np.einsum('jklm,jka->lma',np.tensordot(Gl, V,(0,0)),Gm)
        Tshape = Theta.shape
        #Mtheta is matrix form with index (alpha, i) x (j, gamma)
        Mtheta = Theta.reshape((Tshape[0], Tshape[1]*Tshape[2]))
        u,s,v = LA.svd(Mtheta, full_matrices=False)
        Gl_new = u
        Gm_new = v.reshape(len(s), Tshape[1], Tshape[2])
        recrTheta = np.tensordot(np.tensordot(Gl_new, np.diag(s),(1,0)), Gm_new, (1,0))
    elif l == len(G) - 2:
        Theta = np.einsum('ijklm,jk->ilm',np.tensordot(Gl, V,(1,0)),Gm)
        Tshape = Theta.shape
        #Mtheta is matrix form with index (alpha, i) x (j, gamma)
        Mtheta = Theta.reshape((Tshape[0]*Tshape[1], Tshape[2]))
        u,s,v = LA.svd(Mtheta, full_matrices=False)
        Gl_new = u.reshape(Tshape[0], Tshape[1], len(s))
        Gm_new = v
        recrTheta = np.tensordot(np.tensordot(Gl_new, np.diag(s),(2,0)), Gm_new, (2,0))
    tot = np.sqrt(np.sum(np.conj(s)*s))
    #s = s/tot
    #G[l] = Gl_new*tot
    G[l] = Gl_new
    G[l+1] = Gm_new
    SS[l] = s
    '''
    print(Gl_new.shape, Gm_new.shape)
    print('Norm of Theta is ', np.sqrt(np.sum(np.conj(Theta) * (Theta))))
    print("Difference between Theta and MPS is ", np.sqrt(np.sum(np.conj(recrTheta-Theta) * (recrTheta-Theta))))
    '''
    return G, SS