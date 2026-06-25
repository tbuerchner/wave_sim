import numpy as np

def compute_A_matrices(density, wave_speed, alpha, normals):
    A = np.zeros((2, 2))
    A[0, 1] = 1 / density
    A[1, 0] = density * wave_speed * wave_speed
    A_fict = np.zeros((2, 2))
    A_fict[0, 1] = alpha / density
    A_fict[1, 0] = density * wave_speed * wave_speed

    An = []
    absAn = []
    An_fict = []
    absAn_fict = []
    # Ap = []
    # Am = []

    for n in normals:
        An_now = n[0] * A
        An_fict_now = n[0] * A_fict

        lambda_n_now, Rn_now = np.linalg.eig(An_now)
        invRn_now = np.linalg.inv(Rn_now)
        
        lambda_n_fict_now, Rn_fict_now = np.linalg.eig(An_fict_now)
        invRn_fict_now = np.linalg.inv(Rn_fict_now)

        An.append(An_now)
        absAn.append(Rn_now @ np.diag(np.abs(lambda_n_now)) @ invRn_now)
        An_fict.append(An_fict_now)
        absAn_fict.append(Rn_fict_now @ np.diag(np.abs(lambda_n_fict_now)) @ invRn_fict_now)
        # Ap.append(Rn_now @ np.diag(lambda_n_now * (lambda_n_now > 0)) @ invRn_now)
        # Am.append(Rn_now @ np.diag(lambda_n_now * (lambda_n_now < 0)) @ invRn_now)

    return An, absAn, An_fict, absAn_fict

def compute_rhs(Q_now, mesh_1d, An, absAn, An_fict, absAn_fict, data_M, data_K, data_M_full, data_K_full, dom, mat, alpha):
    rhs = np.zeros_like(Q_now)

    for e in range(mesh_1d.n_elements_total):
        Flux = compute_flux(Q_now, e, mesh_1d, An, absAn, An_fict, absAn_fict, dom, alpha)

        # values at current element
        v  = Q_now[e, :, 0]
        sx = Q_now[e, :, 1]

        # volume contribution
        rhs_v = data_M[e] @ (- (1.0 / mat.density) * data_K[e] @ sx + Flux[:,0])
        rhs_s = data_M_full[e] @ (- mat.bulk_modulus * data_K_full[e] @ v + Flux[:,1])

        rhs[e, :, 0] = rhs_v
        rhs[e, :, 1] = rhs_s

    return rhs

def compute_flux(Q_now, element_index, mesh_1d, An, absAn, An_fict, absAn_fict, dom, alpha):
    # local flux contribution: shape (n_nodes_element_tot, 2)
    flux = np.zeros_like(Q_now[element_index])
    x_bounds_el = mesh_1d.map_to_physical_dim(0, element_index, np.array([-1, 1]))

    # loop over the 2 faces of this element
    for i_face in [0, 1]:        
        if i_face == 0:
            io_node = dom([x_bounds_el[i_face]])
            node_index = 0
            # neighbor element on the left
            if element_index == 0:
                element_index_neighbor = None  # boundary
            else:
                element_index_neighbor = element_index - 1
                node_index_neighbor = mesh_1d.polynomial_degree[0]
        elif i_face == 1:
            node_index = mesh_1d.polynomial_degree[0]
            # neighbor element on the right
            if element_index == (mesh_1d.n_elements[0] - 1):
                element_index_neighbor = None  # boundary
            else:
                element_index_neighbor = element_index + 1
                node_index_neighbor = 0
                
        U_minus = Q_now[element_index, node_index, :]            
        if element_index_neighbor is not None:
            U_plus = Q_now[element_index_neighbor, node_index_neighbor, :]
        else:
            U_plus = np.zeros_like(U_minus)                    
            U_plus[0] = U_minus[0]
            U_plus[1] = -U_minus[1]
        
        if io_node > 0.5:
            flux[node_index, :] = 0.5 * (An[i_face] @ (U_minus + U_plus) + absAn[i_face] @ (U_plus - U_minus))
        else:        
            flux[node_index, :] = 0.5 * (An_fict[i_face] @ (U_minus + U_plus) + absAn_fict[i_face] @ (U_plus - U_minus))

    return flux