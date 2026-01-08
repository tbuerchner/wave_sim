import numpy as np

def gaussian_pulse(x0, sigma, wave_speed=1.0):
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