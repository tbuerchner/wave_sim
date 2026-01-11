import numpy as np

def all_physical():
    """Domain description just reurning 1."""
    def io_test(coords):
        return 1.0
    
    return io_test

def immersed_1d_bar(x_bounds):
    """Create implicit function for an immersed 1D bar."""
    def io_test(coords):
        x = coords[0]
        # Define the bar's boundaries
        x_min, x_max = x_bounds
        # Check if point is inside the bar
        io = np.where((x_min <= x) & (x <= x_max), 1.0, 0.0)
        return io
        
    return io_test
