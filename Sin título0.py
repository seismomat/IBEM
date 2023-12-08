# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:32:27 2023

@author: LLozadaP
"""

#REVISAR LOS VALORES DE G_22 Y g_22


import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy.special as scis
from scipy.special import hankel1, hankel2
from scipy import integrate
from scipy.integrate import fixed_quad


#Hay que revisar todo, hay cosas mal T-T
#Visualización del problema
L=2   #Longitud del dominio
Np=8     #Número de puntos en el dominio
Fx=L/2          #Posición de la fuente (coordenada x)
Fy=-1     #Posición de la fuente (coordenada y)
Vector_fuente=np.array([Fx,Fy])

#Propiedades del medio
f=0.5
w=2*np.pi*f
beta=0.3   #Velocidad de la onda S
Rho=0.3  #Densidad del medio 
mu=(beta**2)*Rho              #Segunda constante de Lamé
k=w/beta     #1/km
k=1
lam=beta/f          #longitud de onda en km
alfa=np.sqrt((lam+(2*mu))/Rho) #Velocidad de la onda P


x=np.linspace(0,L,Np) #Puntos de la "malla"
y=np.zeros(len(x))    #Puntos para graficar el dominio
#Puntos medios
x_m=[]  #Lista vacias para guardar los valores de Xi, puntos medios
for i in range(len(x)-1):
  x_m.append(((x[i+1]-x[i])/2)+x[i])  #Cálculo de los valores de Xi
y_m=np.zeros(len(x_m))  #Vectores para graficar Xi

#Visualización de la geometría
plt.plot(x,y,'k',label='Dominio')
plt.plot(x,y,'|r',label='Divisiones del dominio')
plt.plot(x_m,y_m,'og',label='Receptores')
plt.plot(Fx,Fy,'*b',label='Fuente')
plt.xlim(-0.5,L+0.5)
plt.ylim(-2.5,2.5)
plt.xlabel('Distancia (m)')
plt.ylabel('Desplazamiento (m)')
plt.title('Geometría del problema')
plt.grid()
plt.legend()
plt.show()



#Hasta aquí todo esta bien 

#DEFINICIÓN DE FUNCIONES A EMPLEAR
#Se emplea para el cálculo de T22 (función de Hankel de segunda especie de orden uno)
def d(k,r):
  Di=(k)*r
  #Di=k*r*hankel2(1,k*r)  #Función de Hankel de segundo tipo de primer orden
  return Di

#Se emplea para el cálculo de G22 (función de Hankel de segunda especie de orden cero)
def h(k,r):
  H=hankel2(0,k*r)      #Función de Hankel de segundo tipo de orden cero
  return H

#Delta de Kronecker
def Kron(n,l):
  if n==l:
    return 1
  else:
    return 0


#Cálculo de T_22 "Sistemas de ecuaciones"
Xi=x_m   #Se sobrescribe los valores de Xi
X=x_m
Yi=y_m
Y=y_m
i=1j

#Se establece un vector normal unitario para el dominio
Nx=[0,1]

Dist_X=np.zeros((len(Xi),len(X)))
Dist_Y=np.zeros((len(Xi),len(X)))
r=np.zeros((len(Xi),len(X)))
Gamma=np.zeros((len(Xi),len(X)))
T_22=np.zeros((len(Xi),len(Xi)),dtype=complex)
for n in range(len(Xi)):
  for l in range(len(Xi)):
    #Cálculo de r
    Dist_X[n][l]=X[n]-Xi[l]
    Dist_Y[n][l]=Y[n]-Yi[l]
    r[n][l]=np.sqrt((X[n]-Xi[l])**2+(Y[n]-Yi[l])**2)
    #Cálculo de Gamma
    if n==l:
      Gamma[n][l]=0
    else:
      Gamma[n][l]=(((Dist_X[n][l]/r[n][l])*Nx[0])+((Dist_Y[n][l]/r[n][l])*Nx[1]))       #Este es el producto punto de Gamma por el vector normal a la superficie (respectivo con cada una de sus componentes)
      #T_22=(i*Gamma*Vector_un*d(k,r))/(4*(np.linalg.norm(X[n]-Xi[l])))
      Hankel=d(k,r)
    #Es una matriz
    if n!=l:
      T_22[n][l]=((Hankel[n][l])*(Gamma[n][l])*i)/(4*r[n][l])
    else:
      T_22[n][l]=0

#Los valores de r son correctos

#Integración Gaussiana

T_22_int = np.zeros((len(Xi), len(Xi)), dtype=complex)

for n in range(len(x)-1):
    for l in range(len(x)-1):
        a = x[n]
        b = x[n+1]

        def real_part(x_val):
            return np.real(T_22[n][l])

        def imag_part(x_val):
            return np.imag(T_22[n][l])

        real_integral, _ =integrate.quad(real_part,a,b)
        imag_integral, _ =integrate.quad(imag_part,a,b)

        integral_value=real_integral+1j*imag_integral
        T_22_int[n][l]=integral_value

#Matriz de tracciones (t22)
n=len(x_m)
l=len(x_m)
t_22=np.zeros((n,l),dtype=complex)
for n in range(len(x_m)):
  for l in range(len(x_m)):
    if n==l:
      t_22[n][l]=(0.5)*Kron(n,l)+T_22_int[n][l]
    else:
      t_22[n][l]=Kron(n,l)+T_22_int[n][l]


#CÁLCULO DE LAS TRACCIONES EN SUPERFICIE LIBRE (t_0)
""""En general, el cálculo de las tracciones en superficie
libre es similar al cálculo de t_22"""
#Calculo de las tracciones en superficie libre (t_0)
Coordenadas=np.column_stack((x_m,y_m))  #Coordenadas de los receptores

Distancia_vectores_x=[]  #Vectores desde la fuente hasta cada uno de los receptores en x
Distancia_vectores_y=[]  #Vectores desde la fuente hasta cada uno de los receptores en y
Distancia_vectores_receptores=[]
for i in range(len(x_m)):
  Distancia_vectores_receptores.append(np.sqrt(((x_m[i]-Fx)**2)+((y_m[i]-Fy)**2)))    #Este es el valor de r (para el caso de t_0)
  Distancia_vectores_x.append((x_m[i]-Fx))
  Distancia_vectores_y.append((y_m[i]-Fy))

#Hasta este punto se han generado los valores de r, que son las distancias desde la fuente hasta el receptor
#La variable Distancia_vectores_receptores es igual a r (es un vector con todas las distancias)
#Se hace el producto de los vectores normales a la superficie con los respectivos valores de Gamma

Gamma_Nx=[]
for i in range(len(x_m)):
    Gamma_Nx.append(((Distancia_vectores_x[i]/Distancia_vectores_receptores[i])*Nx[0])+((Distancia_vectores_y[i]/Distancia_vectores_receptores[i])*Nx[1]))

#Hasta este punto se han calculado los productos punto respectivos a los valores de Gamma y Nx

#Ahora se calculan las tracciones T_22_0
T_22_0=[]
Hankel_0=d(k,Distancia_vectores_receptores)
for n in range(len(x_m)):   
    T_22_0.append((((Hankel_0[n])*(Gamma_Nx[n]))/(4*Distancia_vectores_receptores[n]))*1j)
T_22_0=np.array(T_22_0)


T_22_int_0= np.zeros((len(Xi),1), dtype=complex)

for n in range(len(x)-1):
        a = x[n]
        b = x[n+1]

        def real_part(x_val):
            return np.real(T_22_0[n])

        def imag_part(x_val):
            return np.imag(T_22_0[n])

        real_integral, _ =integrate.quad(real_part,a,b)
        imag_integral, _ =integrate.quad(imag_part,a,b)

        integral_value=real_integral+1j*imag_integral
        T_22_int_0[n]=integral_value

t_0=T_22_int_0

#RESOLUCIÓN DEL SISTEMA DE ECUACIONES
t_22_inv=np.linalg.inv((t_22))
phi=t_22_inv@(-t_0)

#Cálculo del campo difractado
#Cálculo de G_22
G_22=np.zeros((len(Xi),len(Xi)),dtype=complex)
Hankel0=h(k,r)
for n in range(len(Xi)):
  for l in range(len(Xi)):
    
    if n!=l:
      G_22[n][l]=(Hankel0[n][l])/(4*Rho*i*beta**2)
    else:
      G_22[n][l]=0


#Cálculo de g_22 (Integración Gaussiana)
g_22=np.zeros((len(Xi),len(Xi)),dtype=complex)
for n in range(len(x)-1):
  for l in range(len(x)-1):
        a=x[n]
        b=x[n+1]

        def real_part(x_val):
            return np.real(G_22[n][l])

        def imag_part(x_val):
            return np.imag(G_22[n][l])

        real_integral, _ =integrate.quad(real_part,a,b)
        imag_integral, _ =integrate.quad(real_part,a,b)

        integral=real_integral+ 1j*imag_integral

        g_22[n][l]=integral

#Campo difractado
u_d=g_22@phi     #indica el producto matricial, en este caso es un vector por una matriz
print(np.real(u_d))

#Visualización de la geometría
plt.plot(x,y,'k',label='Dominio')
plt.plot(x,y,'|r',label='Divisiones del dominio')
plt.plot(x_m,y_m,'og',label='Receptores')
plt.plot(Fx,Fy,'*b',label='Fuente')
plt.plot(x_m,np.real(+u_d),'or',label='Difractado')
plt.xlabel('Distancia (m)')
plt.ylabel('Desplazamiento (m)')
plt.xlim(-0.5,L+0.5)
#plt.ylim(-2.00,2.00)
plt.title('Geometría del problema')
plt.grid()
plt.legend()
plt.show()



plt.plot(phi)
plt.show()
#La respuesta anterior son los desplazamientos en tiempo
#Se obtendrá la respuesta del sistema en frecuencia





