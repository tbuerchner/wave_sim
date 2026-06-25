import numpy as np
import func

def create_location_map(i_element, j_element, n_dof_direction, p):
    dof_indices_i = np.arange(p + 1) + p * i_element
    dof_indices_j = np.arange(p + 1) + p * j_element

    return np.add.outer(dof_indices_j * n_dof_direction[0], dof_indices_i).ravel()

def create_element_slice(i_element, j_element, n_elements, n_dof_element):
    element_index = func.get_element_index(i_element, j_element, n_elements)

    return slice(n_dof_element ** 2 * element_index, n_dof_element ** 2 * (element_index + 1))