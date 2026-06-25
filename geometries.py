import numpy as np

def all_physical():
    """Domain description just reurning 1."""
    def io_test(coords):
        return 1.0
    
    return io_test

def all_fictitious():
    """Domain description just reurning 0."""
    def io_test(coords):
        return 0.0
    
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

def hyperrectangle_with_hypersphere(center, radius):
    """Create implicit function for a hyperrectangle with a hyperspherical hole."""
    def io_test(coords):
        if coords[0].size == 1:
            distance_from_center = np.linalg.norm(coords - center, axis=0)
        else:
            distance_from_center = np.linalg.norm(coords - np.outer(center, np.ones(coords[0].size)), axis=0)
        inside_sphere = distance_from_center <= radius
        io = np.where(~inside_sphere, 1.0, 0.0)
        return io
    
    return io_test

