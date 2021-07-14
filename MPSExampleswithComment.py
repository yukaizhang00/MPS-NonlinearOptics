# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 20:25:36 2021

@author: yukai
"""
import numpy as np
import math
from numpy import linalg as LA
from scipy import linalg as SLA
from MPS_MPO import *

c2 = np.zeros((2,2,2,2,2,2), dtype = np.complex_)
c2[(0,0,0,0,0,0)] += 1.
c2[(1,1,0,0,0,0)] += 1/2.
c2[(0,0,1,1,0,0)] += 1/2.
c2[(0,1,1,0,0,0)] += 1/2.
c2[(0,0,0,0,1,1)] += 1/2.
c2[(0,0,0,1,1,0)] += 1/2.
#c2 += (np.random.rand(2,2,2,2,2,2) + np.random.rand(*(2,2,2,2,2,2)) * 1j)*0.005

c2 = c2/np.sqrt(np.sum(np.conj(c2) * c2))


G,S = MPS(c2)