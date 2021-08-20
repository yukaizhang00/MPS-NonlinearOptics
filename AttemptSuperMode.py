# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 20:59:25 2021

@author: yukai

Attemp for time Evolution
"""

import numpy as np
import math
from numpy import linalg as LA
from scipy import linalg as SLA
from scipy import integrate
from MPS_MPO_fix import *
import matplotlib.pyplot as plt
import pickle


#Attemp for Time evalution
L = 10. #For the total length
N = 80 #For the number of blocks
n = 10 #For the dimension of each block
it = 0
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
    loc = i*L/N - L/2
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
for j in range(1,N-1):
    for i in range(n):
        middle[j-1][(0,i,0)] += np.exp(-(f[j]*alpha)*np.conj(f[j]*alpha)/2)*np.power(np.conj(f[j])*alpha,i)/(np.sqrt(math.factorial(i)))
Gamma = [head] + middle + [tail]
S = [np.array([1. + 0.j]) for i in range(N-1)]




#Notice this is the part for evaluate for t be different value:
    
Uspm = SLA.expm(- 1.j * deltat *SPM)
Udis = SLA.expm(- 1.j * deltat * dispsn.reshape(n*n, n*n)).reshape(n,n,n,n)
for i in range(it):
    print('Working on',i,'th iteration of time evolution')
    #SPM first
    for j in range(N):
        Gamma,S = one_mode(Gamma,S,Uspm,j)
    #Even dispersion part (Notice our index start from 0)
    
    for j in range(1,N-1,2):
        Gamma,S = two_mode(Gamma,S,Udis,j)
    
    for j in range(0,N-1,2):
        Gamma,S = two_mode(Gamma,S,Udis,j)




    

#Reading from the previous result~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
with open('Tevolution500.npy', 'rb') as f:
    #Gevt = np.load(f, allow_pickle=True)
    #Sevt = np.load(f, allow_pickle=True)
    Gamma = np.load(f, allow_pickle=True)
    S = np.load(f, allow_pickle=True)
'''
'''
with open('STevl500.pickle', 'rb') as f2:
    S = pickle.load(f2)[-1]
    
with open('GTevl500.pickle', 'rb') as f2:
    Gamma = pickle.load(f2)[-1]
'''

'''
with open('nbar3chi50fixed8010_0001t1501p100.p', 'rb') as f2:
    Gamma = pickle.load(f2)[0]
    S = pickle.load(f2)[0]
'''
#Gamma = Gevt[-1]
#S = Sevt[-1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Compute super modes:
superf = [f]
c = [np.zeros((N,N), dtype = np.complex_)]
for i in range(N):
    c[0][i][i] += 1.0

phi = []
theta = []
rnum = len(superf)
for r in range(rnum):
    fevo = superf[r]
    cprev = c[-1]
    cnow = np.copy(cprev)
    phinow = np.zeros((N,), dtype = np.complex_)
    thetanow = np.zeros((N,), dtype = np.complex_)
    g = np.zeros((N,), dtype = np.complex_)
    Icur = fevo[0]/cprev[r][0]
    phinow[r] += np.arcsin(np.sqrt(np.conj(Icur)*Icur));thetanow[r] += (np.angle(Icur))
    g[r] += np.exp((1.0j)*thetanow[r]) * np.sin(phinow[r])
    
    #Iteration step
    for m in range(r + 1, N):
        utemp = fevo[m - r]
        for k in range(r,m):
            utemp -= g[k]*cprev[k][m-r]
        btemp = cprev[m][m-r]
        for k in range(r,m):
            btemp *= np.cos(phinow[k])
        Icur = utemp/btemp
        phinow[m] += np.arcsin(np.sqrt(np.conj(Icur)*Icur));thetanow[m] += (np.angle(Icur))
        g[m] += np.exp((1.0j)*thetanow[m]) * np.sin(phinow[m])
        for j in range(r,m):
            g[m] *= np.cos(phinow[j])
    
    #Now we got all Theta^(r), Phi^(r) and g^(r), we want to calculate c^(r) directly
    cnow[N-1] *= np.exp(1.0j*thetanow[N-1])
    for m in range(N-2,r-1,-1):
        cl = np.copy(cnow[m]); cr = np.copy(cnow[m+1])
        cnow[m] = np.exp(1.0j*thetanow[m])*np.sin(phinow[m])*cl + np.cos(phinow[m])*cr
        cnow[m + 1] = np.exp(-1.0j*thetanow[m])*np.sin(phinow[m])*cr - np.cos(phinow[m])*cl
    phi.append(phinow)
    theta.append(thetanow)
    c.append(cnow)

amtam1 = np.zeros((n,n,n,n), dtype = np.complex_)
for i in range(n-1):
    for j in range(1, n):    
        amtam1[(i,j,i+1,j-1)] += (np.sqrt(i+1)*np.sqrt(j))
amtam1 = amtam1.reshape(n*n, n*n)

amtam = np.zeros((n,n,n,n), dtype = np.complex_)
for i in range(n):
    for j in range(n):
        amtam[(i,j,i,j)] += i
amtam = amtam.reshape(n*n, n*n)

am1tam1 = np.zeros((n,n,n,n), dtype = np.complex_)
for i in range(n):
    for j in range(n):
        am1tam1[(i,j,i,j)] += j
am1tam1 = am1tam1.reshape(n*n, n*n)

amam1t = np.zeros((n,n,n,n), dtype = np.complex_)
for i in range(1, n):
    for j in range(n-1):    
        amam1t[(i,j,i-1,j+1)] += (np.sqrt(i)*np.sqrt(j+1))
amam1t = amam1t.reshape(n*n, n*n)

amtamone = np.zeros((n,n), dtype = np.complex_)
for i in range(n):
    amtam[(i,i)] += i

#Now we can get the operator V^(r) as set of two mode operators R^(r)_m:
R = []
RN = []
for r in range(rnum):
    Rnow = []
    for m in range(r,N-1):
        Rm = np.dot(SLA.expm(1.j*theta[r][m] * (amtam -am1tam1)), SLA.expm((np.pi/2 - phi[r][m]) * (np.exp(-1.0j * theta[r][m]) * amtam1 - np.exp(1.0j * theta[r][m])* amam1t)))
        Rnow.append((Rm, m))
    R.append(Rnow)
    RN.append(SLA.expm(1.j * theta[r][N-1] * (amtamone)))

#Calculate the State after V of Gamma and S:
GaftV = [np.copy(gg) for gg in Gamma]
SaftV = [np.copy(ss) for ss in S]

'''
#Testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
recre = GaftV[0]
for i in range(N-1):
    recre = np.tensordot(recre, np.diag(SaftV[i]), (-1,0))
    recre = np.tensordot(recre, GaftV[i+1], (-1,0))
print('The size of recre is', np.sum(np.conj(recre)*recre))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

for r in range(rnum):
    #Rndagger = np.conj(np.transpose(RN[r]))
    Rndagger = RN[r]
    GaftV, SaftV = one_mode(GaftV, SaftV, Rndagger, N-1)
    for Rm in R[r][::-1]:
        print(Rm[1])
        #Rmdagger = np.conj(np.transpose(Rm[0])).reshape(n,n,n,n)
        Rmdagger = Rm[0].reshape(n,n,n,n)
        GaftV, SaftV = two_mode(GaftV, SaftV, Rmdagger, Rm[1])
        """
        #Testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        recre = GaftV[0]
        for i in range(N-1):
            recre = np.tensordot(recre, np.diag(SaftV[i]), (-1,0))
            recre = np.tensordot(recre, GaftV[i+1], (-1,0))
        print('The size of recre is', np.sum(np.conj(recre)*recre))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        """


head = np.zeros((n,1), dtype = np.complex_)
head[(1,0)] += 1
tail = np.zeros((1,n), dtype = np.complex_)
tail[(0,0)] += 1
middle = [np.zeros((1,n,1), dtype = np.complex_) for i in range(1,N-1)]
for j in range(1,N-1):
    middle[j-1][(0,0,0)] += 1
GaftV = [head] + middle + [tail]
SaftV = [np.array([1. + 0.j]) for i in range(N-1)]


#Now let's compute rou_jj' assume there is only one supermode.

#First we contract the Lambda to be Gamma on the left:
GwithoutS = []
for i in range(len(GaftV) - 1):
    GwithoutS.append(np.tensordot(GaftV[i], np.diag(SaftV[i]), (-1,0)))
GwithoutS.append(GaftV[-1])

rou = np.zeros((n, n), dtype=np.complex_)
#Apply the one mode 
Gcomp = [np.tensordot(GwithoutS[0], GwithoutS[0], (0,0))]
for m in range(1,N-1):
    Gcomp.append(np.einsum('aib,cid->acbd', GwithoutS[m],np.conj(GwithoutS[m])))
Gcomp.append(np.einsum('ai,ci->ac', GwithoutS[N-1],np.conj(GwithoutS[N-1])))
        
for j in range(n):
    for jp in range(n):
        print("Working on",(j,jp))
        Ojjp = np.zeros((n, n), dtype=np.complex_)
        Ojjp[(jp,j)] += 1.0
        Gcomp[0] = np.tensordot(np.tensordot(GwithoutS[0], Ojjp, (0,0)), np.conj(GwithoutS[0]), (-1,0))
        temp = Gcomp[0]
        for m in range(1,N-1):
            temp = np.einsum('ac,acbd->bd', temp, Gcomp[m])
            #print(np.max(temp))
        temp = np.einsum('ac,ac', temp, Gcomp[N-1])
        rou[j][jp] += temp
        



#Great we got rou_jj' now we can calculate the Wigner function
gridn = 50
gridx = np.linspace(-5,5,gridn)
gridp = np.linspace(-5,5,gridn)
gridW = np.zeros((gridn,gridn), dtype = np.complex_)
pl = 10; intgn = 100
inty = np.linspace(-pl,pl,intgn)
delty = 2*pl/intgn
for xi in range(len(gridx)):
    x = gridx[xi]
    for pi in range(len(gridp)):
        p = gridp[pi]
        intg = 0
        '''
        #Using normal equidistance to approximate the integral
        for y in inty:
            for j in range(n):
                for jp in range(n):
                    if np.real(np.conj(rou[(j,jp)])*rou[(j,jp)]) > 1e-25:
                        coefj = [0 for i in range(j+1)]
                        coefj[j] += 1
                        phij = 1/(np.power(2,j)*math.factorial(j))*np.exp(-np.power(x+y/2,2))*np.polynomial.hermite.hermval((x+y/2), coefj)
                        
                        coefjp = [0 for i in range(jp+1)]
                        coefjp[jp] += 1
                        phijp = 1/(np.power(2,jp)*math.factorial(jp))*np.exp(-np.power(x-y/2,2))*np.polynomial.hermite.hermval((x-y/2), coefjp)
                        
                        intg += rou[(j,jp)]*np.exp(- 1.0j * p * y) * phij*np.conj(phijp)
        gridW[(xi,pi)] += intg * delty
        '''
        #Using Gaussian quadrature from Scipy
        interr = 0
        def intef(y):
            result = 0
            for j in range(n):
                for jp in range(n):
                    coefj = [0 for i in range(j+1)]
                    coefj[j] += 1
                    coefjp = [0 for i in range(jp+1)]
                    coefjp[jp] += 1
                    phij = (1/np.sqrt((np.power(2,j)*math.factorial(j))))*np.power(np.pi,-1/4)*np.exp(-np.power(x+y/2,2)/2)*np.polynomial.hermite.hermval((x+y/2), coefj)
                    phijp = (1/np.sqrt((np.power(2,jp)*math.factorial(jp))))*np.power(np.pi,-1/4)*np.exp(-np.power(x-y/2,2)/2)*np.polynomial.hermite.hermval((x-y/2), coefjp)
                    result += rou[(j,jp)]*np.exp(-1.j *y*p)*phij*np.conj(phijp)
            return result
        intv, err = integrate.quadrature(intef, -10, 10)
        interr += err
        gridW[(xi,pi)] += intv/(2*np.pi)
    print('x,p has the integral error', interr)
    print("Done the",xi,"th x", gridn, 'in total')




gridWr = np.real(gridW)
plt.pcolormesh(gridx, gridp, gridWr, cmap='RdBu', vmin=-0.3, vmax=0.3)
print('The rou has norm',np.trace(np.tensordot(np.transpose(rou),rou, (-1,0))))
energyW = 0
for xi in range(len(gridx)):
    x = gridx[xi]
    for pi in range(len(gridp)):
        p = gridp[pi]
        energyW += 0.5*(np.power(x,2)+np.power(p,2))*gridW[(xi,pi)]*(10/gridn)*(10/gridn)
#np.einsum('jklm,jka->lma',np.tensordot(Gl, V,(0,0)),Gm)