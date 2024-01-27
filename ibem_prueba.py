# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 12:06:49 2023

@author: jcossc
"""

import scipy.special as scis
import numpy as np
from statistics import mean
from numpy import linalg as la
from matplotlib import pyplot as plt
from math import pi,exp
from math import sqrt as sq
import json
import pandas as pd

#from obspy.core import Trace, Stream
#from numba import jit

def T22(k,x,xi,ni):
    r_ = la.norm(x-xi)
    D = k*r_ * scis.hankel2(1,k*r_)
    return 1j/(4*r_) * D * gam_k__n_k(x,xi,ni)

def G22(k,x,xi,beta,rho):
    r_ = la.norm(x-xi)
    return 1/(1j*4*rho) * scis.hankel2(0,k*r_)/beta**2

def gam_k__n_k(x,xi,ni):
    r_ = la.norm(x-xi)
    return (x[0]-xi[0])/r_ * ni[0] + (x[1]-xi[1])/r_ * ni[1]

T= 2 # seg
N= 32 # armonicos
dt=0.05
f0=1/T
fmax=1/(2*dt)
N=int(fmax/f0)
df=1/(N*dt)
#frec=np.r_[f0:fmax:df]
frec=np.linspace(f0,fmax,N)

w = 2*pi*frec # rad/s
beta = 2500 # km/s

lambds = beta/frec # long de onda en kilometros
ks = w / beta # 1/km
rho = 1200
mus = beta**2 * rho # 2nd Lame constant

dxs=lambds/16

# fuente puntual (antiplana) en
XI = np.array([5000.0,-50.0])
# receptor 
XX =  np.array([[5000.0,-1.0],[4500.0,-1.0],[5500.0,-1.0]])

L=10000;
chunks=np.round(L/dxs).astype(int)

def malla(chunk):
    vert=np.zeros((chunk,2))
    vert[:,0]=np.linspace(0,L,chunk)
    xis=0.5*(vert[1:]+vert[:-1])
    a=vert[1:]-vert[:-1]
    norms=la.norm(a,axis=1)
    nu=np.zeros_like(a)
    nu[:,0]=-a[:,1]/norms
    nu[:,1]=a[:,0]/norms
    return vert,xis,nu

def Int_T22_uniforme_en_xi(n,l,k,vert,xis,norm):
    if n==l:
        return 0.0
    
    gau_i=[-0.774597,0,0.774597]
    w_i=[0.555556,0.888889,0.555556]
    
    b=vert[l,0];a=vert[l+1,0]; 
    xi_x_gau=[(b-a)/2*gau+(a+b)/2 for gau in gau_i]
    
    b=vert[l,1];a=vert[l+1,1]; 
    xi_z_gau=[(b-a)/2*gau+(a+b)/2 for gau in gau_i]
    
    Xi_gau=[[xi_x_gau[i],xi_z_gau[i]] for i in range(len(gau_i))]
    Xi_gau=np.array(Xi_gau)
    
    L=la.norm(vert[l]-vert[n])
    Int=0.0
    for i in range(len(gau_i)):
        Int += L/2 * w_i[i]*T22(k,xis[n],Xi_gau[i],norm[n])
    return Int

def Solver(k,vert,xis,nu):
    # para cada punto de colocacion (renglon)
    #, la integral de las fuentes sobre la superficie
    Mat=np.zeros([len(xis),len(xis)])
    Mat = Mat + 0j*Mat
    for n in range(len(xis)):
        for l in range(len(xis)):
            if n==l:
                Mat[n,l]=0.5+0j
            else:
                Mat[n,l]=Int_T22_uniforme_en_xi(n,l,k,vert,xis,nu)
                
                
    # fuente
    Fue=np.zeros([len(xis),1])
    Fue = Fue + 0j*Fue
    for l in range(len(xis)):
        Fue[l]=T22(k, xis[l], XI, nu[l])
        
    # encontrar amplitudes phi de la densidad de fuerza: phi dS
    phi = np.linalg.solve(Mat,-Fue)
    
    # phis=[abs(phi[i]) for i in range(len(xis))]
    # plt.scatter(xis[:,0],phis,color='black')
    # plt.show()
    
    return phi

def Desplazamiento_Por_frec(data):
    k,vert,xis,nu,XX,beta,rho=data
    phi=Solver(k,vert,xis,nu)
    V = G22(k,XX,XI,beta,rho)# desplazamiento puntual
    for i_xi in range(len(xis)):
        V+=phi[i_xi]*G22(k,XX,xis[i_xi,:],beta,rho)
        # desplazamiento por cada xi

    return V

def Ricker(dt,N):
    tp=0.3; ts=0.5;
    t=np.zeros(N); r=np.zeros(N)
    for i in range(N):
        t[i]=dt*i;
        a=np.pi*(t[i]-ts)/tp;
        a2=a*a;
        r[i]=(a2-1/2)*np.exp(-a2);
    Fr=np.fft.fft(r)
    Fr=Fr[0:round(N/2)]
    plt.plot(t,r)
    plt.title("Espectro Ricker")
    plt.show()
    
    return Fr
    
#@jit(fastmath=True,forceobj=True)
def IBEM(chunks,ks,XX):
    espectro=[]
    for Iele in range(len(ks)):
        vert,xis,nu=malla(chunks[Iele])
        datos=(ks[Iele],vert,xis,nu,XX,beta,rho)
        espectro.append(Desplazamiento_Por_frec(datos))
    # e=np.array(espectro[1:])
    # e=np.conj(e)
    # e=e.tolist()
    # e=e[::-1]
    # f=list(espectro)
    # ff=list(e)
    # ff+=f
    # espectro+=e
        
    # plt.plot(abs(np.array(ff)),'*r')
    # plt.show()
    return np.array(espectro).reshape((len(espectro),))


def signal(chunks,ks,dt,N,XX):
    Sp=IBEM(chunks,ks,XX) # espectro obtenido con IBEM
    FouRic=Ricker(dt,2*N) # espectro del Ricker
    
    conv_Sp_FouRic=Sp*FouRic;
    conj_conv_Sp_FouRic=conv_Sp_FouRic[1:]
    conj_conv_Sp_FouRic=np.conj(conj_conv_Sp_FouRic)
    conj_conv_Sp_FouRic=np.flip(conj_conv_Sp_FouRic)
    Fou_Senal=np.concatenate((conv_Sp_FouRic,conj_conv_Sp_FouRic),axis=0)
    # Cálculo de la IFFT para recuperar la señal en el dominio del tiempo
    # plt.subplot(131)
    # plt.plot(abs(Sp))
    # plt.subplot(132)
    # plt.plot(abs(FouRic))
    # # Ajustar el espaciado entre subgráficos
    # plt.subplot(133)
    # plt.plot(abs(Fou_Senal))
    # plt.tight_layout()
    # plt.show()
    señal_recuperada = np.fft.ifft(Fou_Senal)
    
    return señal_recuperada

señales=[]
datos={}
i=0
for XXi in XX:
    señales.append(signal(chunks,ks,dt,N,XXi))
    datos[str(XXi)]=np.real(señales[i])
    df = pd.DataFrame(datos)
    # Guardar en formato CSV
    file='datos'+str(i)+'.csv'
    df.to_csv(file, index=False)
    i+=1

plt.plot(señales[0],'k-',lw=2)
plt.plot(señales[1],'b-',lw=6)
plt.plot(señales[2],'r-',lw=2)
plt.title("Señal en tiempo")
plt.show()


# Crear un Stream de ObsPy para almacenar las trazas sísmicas
# stream = Stream()

# num_traces=len(señales)
# # Generar trazas sísmicas simuladas
# for i in range(num_traces):

#     # Crear un objeto Trace de ObsPy
#     trace = Trace(data=señales[i])
#     trace.stats.starttime = 0  # Tiempo de inicio de la traza
#     trace.stats.network = "SY"
#     trace.stats.station = f"STATION_{i}"  # Nombre de la estación
#     trace.stats.channel = "HHZ"  # Canal (vertical)

#     # Agregar la traza al Stream
#     stream.append(trace)

# # Graficar las trazas sísmicas
# plt.figure(figsize=(10, 6))
# for i, trace in enumerate(stream):
#     plt.plot(trace.times(), trace.data + i * 2, label=trace.stats.station)

# plt.xlabel('Tiempo (s)')
# plt.ylabel('Amplitud')
# plt.title('Trazas Sísmicas Simuladas')
# plt.legend()
# plt.grid(True)
# plt.show()