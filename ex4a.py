"""
# ex4a.py
#
# Fitting 1st column of the admittance matrix of 6-terminal system
# (power system distribution network)
#
# -Reading frequency admittance matrix Y(s) from disk.
# -Extracting 1st column: f(s) (contains 6 elements)
# -Fitting f(s) using vectfit3.m
#   -Initial poles: 25 linearly spaced complex pairs (N=50)
#   -5 iterations
#
# This example script is part of the vector fitting package (VFIT3.zip)
# Last revised: 08.08.2008.
# Created by:   Bjorn Gustavsen.
#
"""
from vectfit import *
import numpy as np

fid1 = open('fdne.txt', 'r')
Nc = int(fid1.readline())
Ns = int(fid1.readline())

bigY = np.zeros((Nc,Nc,Ns), dtype=np.complex64)
s = np.zeros(Ns, dtype=np.complex64)
for k in range(Ns):
    s[k] = float(fid1.readline())
    for row in range(Nc):
        for col in range(Nc):
            dum1 = float(fid1.readline())
            dum2 = float(fid1.readline())
            bigY[row,col,k] = dum1 + 1j*dum2

fid1.close()
s = 1j*s

#Extracting first column
f = np.zeros((Nc,Ns), dtype=np.complex64)
for n in range(Nc):
    f[n] = bigY[n,1,:]

#=====================================
# Rational function approximation of f(s):
#=====================================
N = 50 #Order of approximation

#Complex starting poles :
w = s/1j
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
                                       auto_rescale=False)
fitted_f = np.zeros(f.shape, dtype=np.complex64)
for i in range(Nc):
    fitted_f[i] = rational_model(s, poles, residues[i], d[i], h[i])

freq = (w/(2*np.pi)/1e3).real
# PLOT
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
#ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("f [kHz]")
ax1.set_ylabel("Magnitude [p.u.]")
ax1.legend(["true", "fitted", "deviation"])
for i in range(Nc):
    ax1.plot(freq, np.abs(f[i]), 'b-')
    ax1.plot(freq, np.abs(fitted_f[i]), 'r-')
    ax1.plot(freq, np.abs(fitted_f[i] - f[i]), 'g--')


fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
#ax2.set_xscale("log")
ax2.set_ylabel("Angle [deg]")
ax2.set_xlabel("f [kHz]")
ax2.legend(["true", "fitted"])
for i in range(Nc):
    ax2.plot(freq, np.unwrap(np.angle(f[i], 'deg'), 180), 'b-')
    ax2.plot(freq, np.unwrap(np.angle(fitted_f[i], 'deg'), 180), 'r-')

plt.show()
