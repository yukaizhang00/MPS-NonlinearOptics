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
n = 9 #For the dimension of each block
it = 0
deltat = 0.001
#Calculate SPM for chi^3 optical
deltaz = L/N
t = 2.0
nbar = 2
alpha = np.sqrt(nbar)


head = np.zeros((n,1), dtype = np.complex_)
cTDHF = np.zeros((n, ), dtype = np.complex_)
for i in range(n):
    temp = np.exp(-nbar/2)*np.exp(1.j *(2*i - nbar)*i*nbar*t/8)*(np.power(alpha, i)/(math.factorial(i)))
    cTDHF[i] += temp
for i in range(n):
    head[(i,0)] += cTDHF[i]



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
        gridW[(xi,pi)] += intv
    print('x,p has the integral error', interr)
    print("Done the",xi,"th x", gridn, 'in total')




gridWr = np.real(gridW)
plt.pcolormesh(gridx, gridp, gridWr, cmap='RdBu', vmin=-1, vmax=1)
print('The rou has norm',np.trace(np.tensordot(np.transpose(rou),rou, (-1,0))))
#np.einsum('jklm,jka->lma',np.tensordot(Gl, V,(0,0)),Gm)
'''
G_tevlt = [Gamma]
S_tevlt = [S]

G_0 = Gamma
S_0 = S

'''

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
    plt.plot(xpoint,ypoint)
    print(ypoint, np.sum(ypoint),'~~~~~~~~~~~~~~~~~~~')

plotMPS(Gamma, S, xpoint)
'''

    #Even dispersion part