"""
 ex2.py

 -Creating an 18th order frequency response f(s) of 2 elements.
 -Fitting f(s) using vectfit3.m
   -Initial poles: 9 linearly spaced complex pairs (N=18)
   -3 iterations

   This example script is a translation to Python from the one that is
   part of the vector fitting package (VFIT3.zip)
 Created by:   Bjorn Gustavsen.
"""
from vectfit import *
import numpy as np

d = 0.2
h = 2e-5
p = np.array([-4500, -41000,
              (-100+1j*5e3), (-100-1j*5e3),
              (-120+1j*15e3), (-120-1j*15e3),
              (-3e3+1j*35e3), (-3e3-1j*35e3),
              (-200+1j*45e3), (-200-1j*45e3),
              (-1500+1j*45e3), (-1500-1j*45e3),
              (-5e2+1j*70e3), (-5e2-1j*70e3),
              (-1e3+1j*73e3), (-1e3-1j*73e3),
              (-2e3+1j*90e3), (-2e3-1j*90e3)
             ], dtype=np.complex64)

r = np.array([-3000, -83000,
              (-5+1j*7e3), (-5-1j*7e3),
              (-20+1j*18e3), (-20-1j*18e3),
              (6e3+1j*45e3), (6e3-1j*45e3),
              (40 +1j*60e3), (40-1j*60e3),
              (90 +1j*10e3), (90-1j*10e3),
              (5e4+1j*80e3), (5e4-1j*80e3),
              (1e3+1j*45e3), (1e3-1j*45e3),
              (-5e3+1j*92e3), (-5e3-1j*92e3)
             ], dtype=np.complex64)

freq = np.linspace(1, 1e5, 100)
w = 2*np.pi*freq
Ns = w.size
s = 1j*w
p = 2*np.pi*p
r = 2*np.pi*r
p1 = p[0:9]
r1 = r[0:9]
N1 = p1.size
p2 = p[8:17]
r2 = r[8:17]
N2 = p2.size
f = np.zeros((2,Ns), dtype=np.complex64)

for k in range(Ns):
    for n in range(N1):
        f[0,k] = f[0,k] + r1[n]/(s[k] - p1[n])

    f[0,k] = f[0,k] + s[k]*h

f[0,:] = f[0,:] + d


for k in range(Ns):
    for n in range(N2):
        f[1,k] = f[1,k] + r2[n]/(s[k] - p2[n])

    f[1,k] = f[1,k] + s[k]*3*h

f[0,:] = f[0,:] + 2*d

#=========================================
# Rational function approximation of f(s):
#=========================================
N = 18 #Order of approximation

#Complex starting poles :
bet = np.linspace(w[0], w[Ns-1], int(N/2))
poles = np.zeros(2*bet.size, dtype=np.complex64)
i = 0
for n in range(bet.size):
    alf = -bet[n]*1e-2
    poles[i] = (alf - 1j*bet[n])
    poles[i + 1] = (alf + 1j*bet[n])
    i += 2

# Real starting poles :
#poles = -np.linspace(w[0], w[Ns-1], N);

Niter = 3
for i in range(Niter):
    poles, residues, d, h = vector_fitting(f, s, initial_poles=poles)


fitted_f = rational_model(s, poles, residues, d, h)

# PLOT
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
#ax1.set_xscale("log")
ax1.set_yscale("log")

ax1.plot(freq/1e3, np.abs(f[0,:]), 'b-')
ax1.plot(freq/1e3, np.abs(fitted_f[0,:]), 'r--')
ax1.plot(freq/1e3, np.abs(fitted_f - f)[0,:], 'g-')

ax1.plot(freq/1e3, np.abs(f[1,:]), 'b-')
ax1.plot(freq/1e3, np.abs(fitted_f[1,:]), 'r--')
ax1.plot(freq/1e3, np.abs(fitted_f - f)[1,:], 'g-')

ax1.set_xlabel("f [kHz]")
ax1.set_ylabel("Magnitude [p.u.]")
ax1.legend(["true", "fitted", "deviation"])
plt.show()
