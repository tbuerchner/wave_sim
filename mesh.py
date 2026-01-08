import numpy as np

class ndRectangle:
    def __init__(self, dim, origin, lengths, n_elements, polynomial_degree):
        self.dim = dim
        if len(origin) != dim:
            raise ValueError("Length of 'origin' must match the spatial dimension 'dim'.")  
        if len(lengths) != dim:
            raise ValueError("Length of 'lengths' must match the spatial dimension 'dim'.")
        self.origin = origin
        self.lengths = lengths
        self.n_elements = n_elements
        self.polynomial_degree = polynomial_degree

    @property
    def dx(self):
        """Calculate the size of each element in the mesh."""

        return self.lengths / self.n_elements
    
    @property
    def n_elements_total(self):
        """Calculate the total number of elements in the mesh."""
        return np.prod(self.n_elements)
    
    @property
    def n_dofs_element(self):
        """Calculate the number of degrees of freedom per element."""
        return np.prod(self.polynomial_degree + 1)
    
    @property
    def element_boundaries(self):   
        """Calculate the boundaries of each element in the mesh."""
        return [np.linspace(self.origin[i], self.origin[i] + self.lengths[i], self.n_elements[i] + 1)
                               for i in range(self.dim)]
    
    @property
    def n_dofs_global_directions(self):
        """Calculate the number of global degrees of freedom in each spatial direction."""
        return self.n_elements * self.polynomial_degree + 1

    @property
    def n_dofs_global(self):
        """Calculate the total number of degrees of freedom in the mesh."""
        return np.prod(self.n_dofs_global_directions)
    
    def get_element_index_total(self, indices):
        """Get the total element index for a given element specified by its indices."""
        if len(indices) != self.dim:
            raise ValueError("Length of 'indices' must match the spatial dimension 'dim'.")

        element_index_total = indices[0]
        n_elements_previous_dirs = self.n_elements[0]
        for i_d in range(1, self.dim):
            element_index_total += indices[i_d] * n_elements_previous_dirs
            n_elements_previous_dirs *= self.n_elements[i_d]
        
        return element_index_total

    def get_location_map(self, indices):
        """Get the global location map for a given element specified by its indices."""
        if len(indices) != self.dim:
            raise ValueError("Length of 'indices' must match the spatial dimension 'dim'.")

        lm = np.arange(self.polynomial_degree[0] + 1) + self.polynomial_degree[0] * indices[0]
        dofs_previous_dirs = self.n_dofs_global_directions[0]
        for i_d in range(1, self.dim):
            dof_indices_id = np.arange(self.polynomial_degree[i_d] + 1) + self.polynomial_degree[i_d] * indices[i_d]
            lm = np.add.outer(dof_indices_id * dofs_previous_dirs, lm).ravel()
            dofs_previous_dirs *= self.n_dofs_global_directions[i_d]
        
        return lm

    def get_element_slice(self, indices):
        """Get the slice for the assembly in the global coo data array."""
        if len(indices) != self.dim:
            raise ValueError("Length of 'indices' must match the spatial dimension 'dim'.")

        element_index_total = self.get_element_index_total(indices)

        return slice(self.n_dofs_element ** 2 * element_index_total, self.n_dofs_element ** 2 * (element_index_total + 1))
    
    def map_to_physical(self, element_indices, local_coords):
        """Map local coordinates to physical coordinates for a given element."""
        if len(element_indices) != self.dim:
            raise ValueError("Length of 'element_indices' must match the spatial dimension 'dim'.")
        if len(local_coords) != self.dim:
            raise ValueError("Length of 'local_coords' must match the spatial dimension 'dim'.")

        physical_coords = np.zeros(self.dim)
        for i_d in range(self.dim):
            x_start = self.origin[i_d] + element_indices[i_d] * self.dx[i_d]
            x_end = x_start + self.dx[i_d]
            physical_coords[i_d] = 0.5 * (x_start + x_end) + 0.5 * (x_end - x_start) * local_coords[i_d]
        
        return physical_coords
    
    def map_to_reference(self, element_indices, physical_coords):
        """Map physical coordinates to local coordinates for a given element."""
        if len(element_indices) != self.dim:
            raise ValueError("Length of 'element_indices' must match the spatial dimension 'dim'.")
        if len(physical_coords) != self.dim:
            raise ValueError("Length of 'physical_coords' must match the spatial dimension 'dim'.")

        local_coords = np.zeros(self.dim)
        for i_d in range(self.dim):
            x_start = self.origin[i_d] + element_indices[i_d] * self.dx[i_d]
            x_end = x_start + self.dx[i_d]
            local_coords[i_d] = (2 * physical_coords[i_d] - (x_start + x_end)) / (x_end - x_start)
        
        return local_coords     
