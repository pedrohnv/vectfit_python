"""
 ex3.py

 Fitting a measured admittance function from distribution transformer
 (single element)

 -Reading frequency response f(s) from disk. (contains 1 element)
 -Fitting f(s) using vectfit3.m
   -Initial poles: 3 linearly spaced complex pairs (N=6)
   -5 iterations

   This example script is a translation to Python from the one that is
   part of the vector fitting package (VFIT3.zip)
 Created by:   Bjorn Gustavsen.
"""
from vectfit import *
import numpy as np

fid1 = open('03PK10.txt', 'r')
line = fid1.readline()
A1 = float(line.split()[0])
A2 = float(line.split()[1])

f = np.zeros(160, dtype=np.complex64)
for k in range(160):
    line = fid1.readline()
    A1 = float(line.split()[0])
    A2 = float(line.split()[1])
    f[k] = A1*np.exp(1j*A2*np.pi/180)

fid1.close()

freq = np.linspace(0, 10e6, 401)
w = 2*np.pi*freq[1:161]
s = 1j*w
Ns = s.size

#=====================================
# Rational function approximation of f(s):
#=====================================
N = 6 #Order of approximation

#Complex starting poles :
bet = np.linspace(w[0], w[Ns-1], int(N/2))
poles = np.zeros(2*bet.size, dtype=np.complex64)
i = 0
for n in range(bet.size):
    alf = -bet[n]*1e-2
    poles[i] = (alf - 1j*bet[n])
    poles[i + 1] = (alf + 1j*bet[n])
    i += 2


Niter = 5
poles, residues, d, h = vector_fitting(f, s, initial_poles=poles, n_iter=Niter,
                                       auto_rescale=True)
fitted_f = rational_model(s, poles, residues, d, h)

# PLOT
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
#ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.plot(w/(2*np.pi)/1e3, np.abs(f), 'b-')
ax1.plot(w/(2*np.pi)/1e3, np.abs(fitted_f), 'r-')
ax1.plot(w/(2*np.pi)/1e3, np.abs(fitted_f - f), 'g--')
ax1.set_xlabel("f [kHz]")
ax1.set_ylabel("Magnitude [p.u.]")
ax1.legend(["true", "fitted", "deviation"])
plt.show()

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
#ax2.set_xscale("log")
ax2.plot(w/(2*np.pi)/1e3, np.angle(f, 'deg'))
ax2.plot(w/(2*np.pi)/1e3, np.angle(fitted_f, 'deg'))
ax2.set_ylabel("Angle [deg]")
ax1.set_xlabel("f [kHz]")
ax2.legend(["true", "fitted"])
plt.show()
