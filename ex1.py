"""
 ex1.py

 Fitting an artificially created frequency response (single element)

 -Creating a 3rd order frequency response f(s)
    -Initial poles: 3 logarithmically spaced real poles
    -1 iteration

 This example script is a translation to Python from the one that is
 part of the vector fitting package (VFIT3.zip)

 Created by:   Bjorn Gustavsen.
"""
from vectfit import *
import numpy as np
#Frequency samples:
Ns = 101
freq = np.logspace(0,4,Ns)
s = 2j*np.pi*freq

true_poles = np.array([-5 + 0j, -100 + 500j, -100 - 500j], dtype=np.complex64)
true_residues = np.array([2, 30 + 40j, 30 - 40j], dtype=np.complex64)
true_d = 0.5
true_h = 0.

true_f = rational_model(s, true_poles, true_residues, true_d, true_h)

#Initial poles for Vector Fitting:
N = 3 #order of approximation
inpoles = -2*np.pi*np.logspace(0,4,N) #Initial poles

poles, residues, d, h = vector_fitting(true_f, s, initial_poles=inpoles)
fitted_f = rational_model(s, poles, residues, d, h)

diff = fitted_f - true_f
Nc = 1
rmserr = np.sqrt( np.sum(np.sum(np.abs( np.square(diff) ))) )/np.sqrt(Nc*Ns)
print("rmserr =", rmserr)

# PLOT
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.plot(freq/1e3, np.abs(true_f))
ax1.plot(freq/1e3, np.abs(fitted_f))
ax1.plot(freq/1e3, np.abs(diff))
ax1.set_xlabel("f [kHz]")
ax1.set_ylabel("Magnitude [p.u.]")
ax1.legend(["true", "fitted", "deviation"])

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.set_xscale("log")
ax2.plot(freq/1e3, np.angle(true_f, 'deg'))
ax2.plot(freq/1e3, np.angle(fitted_f, 'deg'))
ax2.set_ylabel("Angle [deg]")
ax1.set_xlabel("f [kHz]")
ax2.legend(["true", "fitted"])
plt.show()
