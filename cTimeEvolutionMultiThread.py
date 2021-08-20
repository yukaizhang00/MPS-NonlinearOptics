# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 20:59:25 2021

@author: yukai

Attemp for time Evolution
"""

import numpy as np
import math
from numpy import linalg as LA
from scipy import linalg as SLA
from MPS_MPO_fix import *
import matplotlib.pyplot as plt
import pickle
import concurrent.futures
import time

st = time.time()


#Attemp for Time evalution
L = 10. #For the total length
N = 80 #For the number of blocks
n = 10 #For the dimension of each block
it = 50
it = it + 1
deltat = 0.001
#Calculate SPM for chi^3 optical
deltaz = L/N
SPM = np.zeros((n,n), dtype = np.complex_)
for i in range(n):
    SPM[(i,i)] += (i/(np.power(deltaz,2)) - i*(i-1)/(2*deltaz))
#dispsn is Dispersion term with index i,j,i',j'
dispsn = np.zeros((n,n,n,n), dtype = np.complex_)
for i in range(1, n-1):
    for j in range(1, n):    
        dispsn[(i,j,i+1,j-1)] += (-(np.sqrt(i+1)*np.sqrt(j))/(2*deltaz*deltaz))
    for j in range(n-1):
        dispsn[(i,j,i-1,j+1)] += (-(np.sqrt(i)*np.sqrt(j+1))/(2*deltaz*deltaz))
for j in range(1,n):
    dispsn[(0,j,1,j-1)] += (-np.sqrt(j)/(2*deltaz*deltaz))
for j in range(n-1):
    dispsn[(n-1,j,n-2,j+1)] += (-(np.sqrt(n-1)*np.sqrt(j+1))/(2*deltaz*deltaz))

#Consider the Soliton initial solution
#nbar is average number of photons
nbar = 2.
alpha = np.sqrt(nbar)
f = []
xpoint = []
for i in range(N):
    loc = (i+1/2)*L/N - L/2
    xpoint.append(loc)
    f.append((nbar/2)/(np.cosh(loc*nbar/2)))
f = np.array(f)
fmnorm = (np.sqrt(np.sum(np.power(f,2))))
f = f/(np.sqrt(np.sum(np.power(f,2))))
print('fm has norm', np.sum(np.power(f,2)),f)
head = np.zeros((n,1), dtype = np.complex_)
for i in range(n):
    head[(i,0)] += np.exp(-(f[0]*alpha)*np.conj(f[0]*alpha)/2)*np.power(np.conj(f[0])*alpha,i)/(np.sqrt(math.factorial(i)))
tail = np.zeros((1,n), dtype = np.complex_)
for i in range(n):
    tail[(0,i)] += np.exp(-(f[-1]*alpha)*np.conj(f[-1]*alpha)/2)*np.power(np.conj(f[-1])*alpha,i)/(np.sqrt(math.factorial(i)))
middle = [np.zeros((1,n,1), dtype = np.complex_) for i in range(1,N-1)]
for j in range(N-2):
    for i in range(n):
        middle[j][(0,i,0)] += np.exp(-(f[j+1]*alpha)*np.conj(f[j+1]*alpha)/2)*np.power(np.conj(f[j+1])*alpha,i)/(np.sqrt(math.factorial(i)))
Gamma = [head] + middle + [tail]
for i in range(len(Gamma)):
    sqs = np.sqrt(np.sum(np.conj(Gamma[i])*Gamma[i]))
    Gamma[i] = Gamma[i]/sqs
S = [np.array([1. + 0.j]) for i in range(N-1)]
'''
for i in range(len(S)):
    S[i] = S[i]/np.sqrt(np.sum(np.power(S[i],2)))
'''
jointloc = 0


'''
rdfname = 'fixed8010_0001t1000p100.p'
with open(rdfname, 'rb') as f2:
    Gamma = pickle.load(f2)[-1]
    S = pickle.load(f2)[-1]

jointloc = rdfname[:-2].split('t')[1].split('p')[0]
'''

G_tevlt = [Gamma]
S_tevlt = [S]


G_0 = [np.copy(g) for g in G_tevlt[-1]]
S_0 = [np.copy(s) for s in S_tevlt[-1]]

#Trying plot the Envelope of initial state:
'''
xp = np.linspace(-deltaz/2, deltaz/2, 100)
yp = xp - xp
for i in range(len(Gamma[0])):
    coe = Gamma[0][i][0]
    for xind in range(len(xp)):
        #yp[xind] += coe * np.sin(np.pi*(i+1)*(xp[xind] + deltaz/2)/deltaz)
        #yp[xind] += coe.real * i
        yp[xind] += (coe* np.conj(coe)).real * i

plt.plot(xp - deltaz*N/2,yp)
for b in range(1,N-1):
    xp = np.linspace(-deltaz/2, deltaz/2, 100)
    yp = xp - xp
    for i in range(len(Gamma[b][0])):
        coe = Gamma[b][0][i][0]
        for xind in range(len(xp)):
            #yp[xind] += coe * np.sin(np.pi*(i+1)*(xp[xind] + deltaz/2)/deltaz)
            #yp[xind] += coe.real * i
            yp[xind] += (coe* np.conj(coe)).real * i
    plt.plot(xp + deltaz*(b-N/2),yp)
    print('=============',max(yp))

xp = np.linspace(-deltaz/2, deltaz/2, 100)
yp = xp - xp
for i in range(len(Gamma[N-1][0])):
    coe = Gamma[N-1][0][i]
    for xind in range(len(xp)):
        #yp[xind] += coe * np.sin(np.pi*(i+1)*(xp[xind] + deltaz/2)/deltaz)
        #yp[xind] += coe.real * i
        yp[xind] += (coe* np.conj(coe)).real * i
plt.plot(xp + deltaz*(N/2-1),yp)


xx = np.linspace(-L/2, L/2, 100)
fint = 0.0
xxxp = np.linspace(-5,5,100)
for i in xxxp:
    fint += 0.1* (nbar/2)/(np.cosh(loc*nbar/2))

plt.plot(xx, (nbar/2)/(np.cosh(xx*nbar/2))*fint)
'''





xpoint = np.array(xpoint)

def plotMPS(Gamma, S, xpoint):
    ypoint = xpoint-xpoint
    #Gammatemp = np.tensordot(np.tensordot(Gamma[0], np.diag(S[0]), (-1,0)), np.diag(np.array(range(n))),(0,0))
    Gammatemp = np.tensordot(Gamma[0], np.diag(S[0]), (-1,0))
    #ypoint[0] += np.sum(np.tensordot(np.real(Gammatemp*np.conj(Gammatemp)), np.diag(np.array(range(n))),(0,0)))
    ypoint[0] += np.sum(np.tensordot(np.real(Gammatemp*np.conj(Gammatemp)), np.diag(np.array(range(n))),(0,0)))
    
    for i in range(1,N-1):
        #Gammatemp = np.tensordot(np.tensordot(np.diag(S[i-1]), np.tensordot(Gamma[i], np.diag(S[i]), (-1,0)), (-1,0)), np.diag(np.array(range(n))),(1,0))
        Gammatemp = np.tensordot(np.diag(S[i-1]), np.tensordot(Gamma[i], np.diag(S[i]), (-1,0)), (-1,0))
        #ypoint[i] += np.sum(np.tensordot(np.real(Gammatemp*np.conj(Gammatemp)), np.diag(np.array(range(n))),(1,0)))
        ypoint[i] += np.sum(np.tensordot(np.real(Gammatemp*np.conj(Gammatemp)), np.diag(np.array(range(n))),(1,0)))
    
    #Gammatemp = np.tensordot(np.tensordot(np.diag(S[N-2]), Gamma[N-1], (-1,0)), np.diag(np.array(range(n))),(1,0))
    Gammatemp = np.tensordot(np.diag(S[N-2]), Gamma[N-1], (-1,0))
    #ypoint[N-1] += np.sum(Gammatemp*np.conj(Gammatemp))
    #ypoint[N-1] += np.sum(np.tensordot(np.real(Gammatemp*np.conj(Gammatemp)), np.diag(np.array(range(n))),(1,0)))
    ypoint[N-1] += np.sum(np.tensordot(np.real(Gammatemp*np.conj(Gammatemp)), np.diag(np.array(range(n))),(1,0)))
    #plt.plot(xpoint,ypoint*10/sum(ypoint))
    #plt.plot(xpoint,ypoint/np.sum(ypoint))
    plt.plot(xpoint,ypoint)
    print(ypoint, np.sum(ypoint),'~~~~~~~~~~~~~~~~~~~')

plotMPS(Gamma, S, xpoint)

Uspm = SLA.expm(- 1.j * deltat *SPM)
Udis = SLA.expm(- 1.j * deltat * dispsn.reshape(n*n, n*n)).reshape(n,n,n,n)
Gtemp = [np.copy(g) for g in G_tevlt[-1]]
Stemp = [np.copy(s) for s in S_tevlt[-1]]

for i in range(it):
    print('Working on iteration', i)
    '''
    Gtemp = [np.copy(g) for g in G_tevlt[-1]]
    Stemp = [np.copy(s) for s in S_tevlt[-1]]
    '''
    #SPM first
    for j in range(N):
        Gtemp,Stemp = one_mode(Gtemp,Stemp,Uspm,j)
    #Even dispersion part (Notice our index start from 0)
    ttemp = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(f, range(N), range(N))
    for j in range(0, N - 1, 2):
        t = threading.Thread(target=two_mode_o, args=[Gtemp, Stemp, Udis, j])
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print('The even ones takes', time.perf_counter() - ttemp)
    threads = []
    for j in range(1, N - 1, 2):
        t = threading.Thread(target=two_mode_o, args=[Gtemp, Stemp, Udis, j])
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    '''
    for j in range(N-1):
        Gtemp,Stemp = two_mode(Gtemp,Stemp,Udis,j)
    '''
    '''
    #Normalizing S for good ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(len(Stemp)):
        Stemp[i] = Stemp[i]/np.sqrt(np.sum(np.power(Stemp[i],2)))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    '''
    #Testing part don't run~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    recreat = Gtemp[0]
    for i in range(1, N):
        recreat = np.tensordot(recreat, np.diag(Stemp[i-1]), (-1,0))
        recreat = np.tensordot(recreat, Gtemp[i], (-1,0))
    print('The c norm is',np.sqrt(np.sum(np.conj(recreat) * recreat)))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    '''
    '''
    G_tevlt.append(Gtemp)
    S_tevlt.append(Stemp)
    '''
    if i % 100 == 0:
        print('Will plot', i)    
        G_tevlt.append(Gtemp)
        S_tevlt.append(Stemp)
        plotMPS(Gtemp, Stemp, xpoint)
    #Even dispersion part
    plt.savefig('Test'+str(N)+str(n)+str(deltat)+'.png')
'''
#fname = '60fixed'+str(N)+str(n)+'_'+str(deltat).replace('.', '')+'j'+str(jointloc)+'t'+str(it)+'p100'+'.p'
fname = 'c50fixed'+str(N)+str(n)+'_'+str(deltat).replace('.', '')+'t'+str(it)+'p100'+'.p'
f = open(fname, 'wb')
pickle.dump(G_tevlt,f)
pickle.dump(S_tevlt,f)
f.close()
'''
print('after time',time.time()-st)