# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:29:18 2024

@author: jcossc
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy import linalg as la

chunk=50

x=np.linspace(-5,5,chunk)
f=lambda x,sigma:np.exp(-x**2/(2*sigma))
#f=lambda x,sigma:(1/(np.sqrt(2*sigma)))*np.exp(-x**2/(2*sigma))
y=-f(x,10)


vert=np.zeros((chunk,2))
vert[:,0]=x
vert[:,1]=y

xis=0.5*(vert[1:]+vert[:-1])
a=vert[1:]-vert[:-1]
norms=la.norm(a,axis=1)
nu=np.zeros_like(a)
nu[:,0]=-a[:,1]/norms
nu[:,1]=a[:,0]/norms

plt.plot(vert[:,0],vert[:,1],'ob')
plt.plot(xis[:,0],xis[:,1],'xr')
plt.show()

plt.plot(vert[:,0],vert[:,1],'--rs')
plt.plot(xis[:,0],xis[:,1],'b*')
for i in range(chunk-1):
    plt.quiver(xis[i,0],xis[i,1],nu[i,0],nu[i,1])
plt.show()