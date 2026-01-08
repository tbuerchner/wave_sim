import numpy as np
import scipy

def get_lagrange_functions(interpolation_points, derivative_order=0):
    """
    Generate the ith derivative of Lagrange functions interpolating at given points.

    Parameters:
    interpolation_points : array-like
        The points at which the Lagrange basis functions are defined.
    derivative_order : int, optional
        The order of the derivative to compute (default is 0, which means no derivative).

    Returns:
    lagrange_funcs : list of callable
        A list of functions representing the Lagrange shape functions ith derivatives.
    """
    n = len(interpolation_points)
    lagrange_values = np.identity(n)

    shapes = [scipy.interpolate.lagrange(interpolation_points, lagrange_values[i]) for i in range(n)]
    if derivative_order == 0:
        lagrange_funcs = shapes
    else:
        lagrange_funcs = [np.polyder(shape, derivative_order) for shape in shapes]
    
    return lagrange_funcs

def get_integrated_Legrende_functions(p, derivative_order=0):
    """
    Generate the ith derivative of integrated Legendre functions up to degree p.

    Parameters:
    p : int
        The maximum degree of the integrated Legendre polynomials.
    derivative_order : int, optional
        The order of the derivative to compute (default is 0, which means no derivative).

    Returns:
    integrated_legendre_funcs : list of callable
        A list of functions representing the integrated Legendre polynomials ith derivatives.
    """
    int_legendre_funcs = []

    for i in range(p + 1):
        if i == 0:
            shape_diff_0 = scipy.interpolate.lagrange([-1, 1], np.array([1, 0]))
            if derivative_order == 0:
                int_legendre_funcs.append(shape_diff_0)
            elif derivative_order == 1:
                int_legendre_funcs.append(np.polyder(shape_diff_0))
            else:
                raise ValueError("Derivative order higher than 1 is not implemented.")
        elif i == p:
            shape_diff_0 = scipy.interpolate.lagrange([-1, 1], np.array([0, 1]))
            if derivative_order == 0:
                int_legendre_funcs.append(shape_diff_0)
            elif derivative_order == 1:
                int_legendre_funcs.append(np.polyder(shape_diff_0))
            else:
                raise ValueError("Derivative order higher than 1 is not implemented.")
        else:
            cs = [0] * (i + 1)
            cs[i] = 1
            cs_scaled = [np.sqrt((2 * i + 1) / 2) * c_now for c_now in cs]
            cs_int = np.polynomial.legendre.legint(cs, lbnd=-1.0)
            cs_int_scaled = [np.sqrt((2 * i + 1) / 2) * c_now for c_now in cs_int]
            if derivative_order == 0:
                int_legendre_funcs.append(np.polynomial.legendre.Legendre(cs_int_scaled))
            elif derivative_order == 1:
                int_legendre_funcs.append(np.polynomial.legendre.Legendre(cs_scaled))
            else:
                raise ValueError("Derivative order higher than 1 is not implemented.")

    return int_legendre_funcs

