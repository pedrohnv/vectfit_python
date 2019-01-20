# -*- coding: utf-8 -*-
"""
@authors: Phil Reinhold, Pedro H. N. Vieira and Alexis Morvan

Duplication of the vector fitting algorithm in python
(http://www.sintef.no/Projectweb/VECTFIT/)

All credit goes to Bjorn Gustavsen for his MATLAB implementation,
and the following papers:

 [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
     vol. 14, no. 3, pp. 1052-1061, July 1999.

 [2] B. Gustavsen, "Improving the pole relocating properties of vector
     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
     July 2006.

 [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
     "Macromodeling of Multiport Systems Using a Fast Implementation of
     the Vector Fitting Method", IEEE Microwave and Wireless Components
     Letters, vol. 18, no. 6, pp. 383-385, June 2008.

A warning about Ill conditioning of the problem may arise. To ignore
it in your code use

```
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fitted = vector_fitting(f,s)
```
"""
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import warnings


def rational_model(s, poles, residues, d, h):
    """
    Complex rational function.

    Parameters
    ----------
    s : array of complex frequencies.
    poles : array of the pn
    residues : array of the rn
    d : real, offset
    h : real, slope

    Returns
    -------
     N
    ----
    \       rn
     \   ------- + d + s*h
     /    s - pn
    /
    ----
    n=1
    """
    f = lambda x: (residues/(x - poles)).sum() + d + np.dot(x, h)
    y = np.vectorize(f)
    return y(s)


def flag_poles(poles, Ns):
    """
    Identifies a given pole:
        0 : real
        1 : complex
        2 : complex.conjugate()

    Parameters
    ----------
    poles : initial poles guess
        note: All complex poles must come in sequential complex
        conjugate pairs
    Ns : number of samples being used (s.size)

    Returns
    -------
    cindex : identifying array
    """
    N = len(poles)
    cindex = np.zeros(N)
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert poles[i].conjugate() == poles[i+1], (
                        "Complex poles"" must come in conjugate "
                        +"pairs: %s, %s" % (poles[i], poles[i+1]))
                cindex[i] = 1
            else:
                cindex[i] = 2

    return cindex


def residues_equation(f, s, poles, cindex, sigma_residues=True,
                      asymptote='affine'):
    """
    Builds the first linear equation to solve. See Appendix A.

    Parameters
    ----------
    f : array of the complex data to fit
    s : complex sampling points of f
    poles : initial poles guess
        note: All complex poles must come in sequential complex
        conjugate pairs
    cindex : identifying array of the poles (real or complex)
    f_residues : bool, default=True
        signals if the residues of sigma (True) or f (False) are being
        calculated. The equation is a bit different in each case.
    asymptote : define the asymptotic behavior to be fitted :
        None, 'linear', 'affine' or 'constant'
    Returns
    -------
    A, b : of the equation Ax = b
    """
    try:
        Ns, Ndim = np.shape(f)
    except ValueError:
        Ns = s.size
        Ndim = 1
    N = poles.size
    A0_list = []
    A1_list = []
    for k in range(Ndim):
        A0 = np.zeros((Ns, N), dtype=np.complex128)
        A1 = np.zeros((Ns, N), dtype=np.complex128)
        for i, p in enumerate(poles):
            if cindex[i] == 0:
                A0[:, i] = 1/(s - p)
            elif cindex[i] == 1:
                A0[:, i] = 1/(s - p) + 1/(s - p.conjugate())
            elif cindex[i] == 2:
                A0[:, i] = 1j/(s - p) - 1j/(s - p.conjugate())
            else:
                raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))

            if sigma_residues:
                if Ndim == 1:
                    A1[:, i] = -A0[:, i]*f
                else:
                    A1[:, i] = -A0[:, i]*f[:, k]
        if asymptote == 'constant':
            A0 = np.concatenate([A0, np.transpose([np.ones(Ns)])], axis=1)
        if asymptote == 'affine':
            A0 = np.concatenate([A0, np.transpose([np.ones(Ns)]),
                                 np.transpose([s])], axis=1)
        if asymptote == 'linear':
            A0 = np.concatenate([A0, np.transpose([s])], axis=1)
        A0_list.append(A0)
        A1_list.append(A1)
    A = np.concatenate([block_diag(*A0_list),
                       np.concatenate(A1_list, axis=0)], axis=1)
    if Ndim == 1:
        b = f
    else:
        b = np.hstack([f[:, k] for k in range(Ndim)])
    A = np.vstack((A.real, A.imag))
    b = np.concatenate((b.real, b.imag))
    cA = np.linalg.cond(A)
    if cA > 1e13:
        message = ('Ill Conditioned Matrix. Cond(A) = ' + str(cA)
                   + ' . Consider scaling the problem down.')
        warnings.warn(message, UserWarning)
    return A, b


def fitting_poles(f, s, poles, asymptote='affine'):
    """
    Calculates the poles of the fitting function.

    Parameters
    ----------
    f : array of the complex data to fit
    s : complex sampling points of f
    poles : initial poles guess
        note: All complex poles must come in sequential complex
        conjugate pairs

    Returns
    -------
    new_poles : adjusted poles
    """
    N = len(poles)
    Ns = len(s)
    cindex = flag_poles(poles, Ns)

    # calculates the residues of sigma
    A, b = residues_equation(f, s, poles, cindex)
    # Solve Ax == b using pseudo-inverse
    x, residues, rnk, s = np.linalg.lstsq(A, b, rcond=-1)

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros of sigma, which are equal to the poles
    # of the fitting function: Appendix B
    A = np.diag(poles)
    b = np.ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = p.real, p.imag
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0
            #cv = c[i]
            #c[i,i+1] = real(cv), imag(cv)

    H = A - np.outer(b, c)
    H = H.real
    eig = np.linalg.eigvals(H)
    new_poles = np.sort(eig)
    unstable = new_poles.real > 0
    new_poles[unstable] -= 2*new_poles.real[unstable]
    return new_poles


def fitting_residues(f, s, poles, asymptote='affine'):
    """
    Calculates the residues of the fitting function.

    Parameters
    ----------
    f : array of the complex data to fit
    s : complex sampling points of f
    poles : calculated poles (by fitting_poles)
    asymptote : shape of the asymptote : affine, linear, constant or None

    Returns
    -------
    residues : adjusted residues
    d : adjusted offset
    h : adjusted slope
    """
    try:
        Ns, Ndim = np.shape(f)
    except ValueError:
        Ns = len(s)
        Ndim = 1
    print(Ndim)
    N = len(poles)
    cindex = flag_poles(poles, Ns)

    if asymptote is None:
        Ntab = N
    elif asymptote == 'affine':
        Ntab = N+2
    elif asymptote == 'linear' or asymptote == 'constant':
        Ntab = N+1

    # calculates the residues of sigma
    A, b = residues_equation(f, s, poles, cindex, False, asymptote=asymptote)
    # Solve Ax == b using pseudo-inverse
    x, residues, _, _ = np.linalg.lstsq(A, b, rcond=-1)
    # Recover complex values
    x = np.complex128(x)
    for i, ci in enumerate(cindex):
        if ci == 1:
            for k in range(Ndim):
                r1, r2 = x[Ntab*k+i:Ntab*k+i+2]
                x[Ntab*k+i] = r1 - 1j*r2
                x[Ntab*k+i+1] = r1 + 1j*r2

    residues = np.squeeze([x[j*Ntab:j*Ntab+N] for j in range(Ndim)])
    if asymptote == 'affine':
        d = [x[Ntab*j + N].real for j in range(Ndim)]
        h = [x[Ntab*j + N+1].real for j in range(Ndim)]
    elif asymptote == 'linear':
        d = [0.]*Ndim
        h = [x[Ntab*j + N].real for j in range(Ndim)]
    elif asymptote == 'constant':
        d = [x[Ntab*j + N].real for j in range(Ndim)]
        h = [0.]*Ndim
    elif asymptote is None:
        d = [0.]*Ndim
        h = [0.]*Ndim
    return residues, d, h


def vector_fitting(f, s, poles_pairs=10, loss_ratio=0.01, n_iter=3,
                   initial_poles=None, asymptote='affine'):
    """
    Makes the vector fitting of a complex function.

    Parameters
    ----------
    f : array of the complex data to fit
    s : complex sampling points of f
    poles_pairs : optional int, default=10
        number of conjugate pairs of the fitting function.
        Only used if initial_poles=None
    loss_ratio : optional float, default=0.01
        The initial poles guess, if not given, are estimated as
        w*(-loss_ratio + 1j)
    n_iter : optional int, default=3
        number of iterations to do when calculating the poles, i.e.,
        consecutive pole fitting
    initial_poles : optional array, default=None
        The initial pole guess

    Returns
    -------
    poles : adjusted poles
    residues : adjusted residues
    d : adjusted offset
    h : adjusted slope
    """
    w = s.imag
    if initial_poles is None:
        beta = np.linspace(w[0], w[-1], poles_pairs+2)[1:-1]
        initial_poles = np.array([])
        p = np.array([[-loss_ratio + 1j], [-loss_ratio - 1j]])
        for b in beta:
            initial_poles = np.append(initial_poles, p*b)

    poles = initial_poles
    for _ in range(n_iter):
        poles = fitting_poles(f, s, poles, asymptote=asymptote)

    residues, d, h = fitting_residues(f, s, poles, asymptote=asymptote)
    return poles, residues, d, h

if __name__ == '__main__':
    #example case
    true_poles = np.array([-4500, -41e3,
                           -100 + 5e3j, -100 - 5e3j,
                           -120 + 15e3j, -120 - 15e3j,
                           -3e3 + 35e3j, -3e3 - 35e3j,
                           -200 + 45e3j, -200 - 45e3j,
                           -15e2 + 45e3j, -15e2 - 45e3j,
                           -500 + 70e3j, -500 - 70e3j,
                           -1e3 + 73e3j, -1e3 - 73e3j,
                           -2e3 + 90e3j, -2e3 - 90e3j],
                          dtype=np.complex128)
    true_residues = np.array([-3e3, -83e3,
                               -5 + 7e3j, -5 - 7e3j,
                               -20 + 18e3j, -20 - 18e3j,
                               6e3 + 45e3j, 6e3 - 45e3j,
                               40 + 60e3j, 40 - 60e3j,
                               90 + 10e3j, 90 - 10e3j,
                               50e3 + 80e3j, 50e3 - 80e3j,
                               1e3 + 45e3j, 1e3 - 45e3j,
                               -5e3 + 92e3j, -5e3 - 92e3j],
                              dtype=np.complex128)
    #true_residues = true_residues.real + true_residues.imag*2*np.pi
    #true_poles = true_poles.real + true_poles.imag*2*np.pi
    true_d = 0.2
    true_h = 2e-5

    freq = np.logspace(0, 5, 200)
    s = 2j*np.pi*freq
    true_f = rational_model(s, true_poles, true_residues, true_d, true_h)

    initial_poles = np.array([-1e-2 + 1j, -1e-2 - 1j,
                              -1.11e2 + 1.11e4j, -1.11e2 - 1.11e4j,
                              -2.22e2 + 2.22e4j, -2.22e2 - 2.22e4j,
                              -3.33e2 + 3.33e4j, -3.33e2 - 3.33e4j,
                              -4.44e2 + 4.44e4j, -4.44e2 - 4.44e4j,
                              -5.55e2 + 5.55e4j, -5.55e2 - 5.55e4j,
                              -6.66e2 + 6.66e4j, -6.66e2 - 6.66e4j,
                              -7.77e2 + 7.77e4j, -7.77e2 - 7.77e4j,
                              -8.88e2 + 8.88e4j, -8.88e2 - 8.88e4j,
                              -1e3 + 1e5j, -1e3 - 1e5j],
                             dtype=np.complex128)

#    poles = fitting_poles(test_f, s, initial_poles)
#    residues, d, h = fitting_residues(test_f, s, poles)
#    fitted = rational_model(s, poles, residues, d, h)
    poles, residues, d, h = vector_fitting(true_f, s, asymptote=None)
    fitted_f = rational_model(s, poles, residues, d, h)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.plot(freq/1e3, np.abs(true_f))
    ax.plot(freq/1e3, np.abs(fitted_f), '--')
    ax.set_xlabel("f [kHz]")
    ax.set_ylabel("Magnitude [p.u.]")
    ax.legend(["true", "fitted"])
    plt.show()
