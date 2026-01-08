import numpy as np

def gaussian_pulse_1d(x0, sigma, wave_speed=1.0):
    """
    Generate a Gaussian pulse centered at x0 with standard deviation sigma.

    Parameters:
    x0 : float
        The center of the Gaussian pulse.
    sigma : float
        The standard deviation of the Gaussian pulse.

    Returns:
    f : lambda function
        A lambda function f(x, t) that returns the value of the Gaussian pulse at position x and time t.
    """

    def f(x, t):
        return np.exp(-(x + wave_speed*t)**2 / (2 * sigma ** 2)) + np.exp(-(x - wave_speed*t)**2 / (2 * sigma ** 2))
    
    return f

def gaussian_pulse_nd(x0, sigma):
    """
    Generate an anisotropic Gaussian pulse in n dimensions.
    
    Parameters:
    x0 : np.array
        Center (nD array).
    sigma : np.array or float
        Standard deviations (array for anisotropic, float for isotropic).
    
    Returns:
    f : lambda function
        f(coords) for the pulse value.
    """
    sigma = np.array(sigma)
    def f(coords):
        diff = coords - x0
        exponent = np.sum(diff**2 / (2 * sigma**2), axis=-1)
        return np.exp(-exponent) 
    
    return f
