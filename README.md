Duplication of the [Fast Relaxed Vector-Fitting algorithm](http://www.sintef.no/Projectweb/VECTFIT/) in python.

This version was inspired by the one made by Phil Reinhold. The changes are mainly of documentation and code organization. There is no auto rescaling of the problem!

Example of use:
```
def vector_fitting(f, s, poles_pairs=10, loss_ratio=0.01, n_iter=3,
                   initial_poles=None):
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
        number of iterations to do to calculate the poles, i.e.,
        consecutive pole fitting
    initial_poles : optional array, default=None
        The initial pole guess
    
    Returns
    -------
    fitted(s) : the fitted function with 's' as parameter
    """
	
f = some_data
s = sampling_points
fitted = vector_fitting(f, s) # returns a lambda function of 's'
fitted_s = fitted(s)    
```
