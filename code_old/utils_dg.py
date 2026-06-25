import numpy as np

import func

def create_location_map(i_element, j_element, n_nodes_direction, p):
    node_indices_i = np.arange(p + 1) + (p + 1) * i_element
    node_indices_j = np.arange(p + 1) + (p + 1) * j_element

    return np.add.outer(node_indices_j * n_nodes_direction[0], node_indices_i).ravel()

def compute_A_matrices(rho, c, normals):
    A = np.zeros((3, 3))
    A[0, 1] = 1 / rho
    A[1, 0] = rho * c * c

    B = np.zeros((3, 3))
    B[0, 2] = 1 / rho
    B[2, 0] = rho * c * c

    An = []
    absAn = []
    Ap = []
    Am = []

    for n in normals:
        An_now = n[0] * A + n[1] * B
        lambda_n_now, Rn_now = np.linalg.eig(An_now)
        invRn_now = np.linalg.inv(Rn_now)

        An.append(An_now)
        absAn.append(Rn_now @ np.diag(np.abs(lambda_n_now)) @ invRn_now)
        Ap.append(Rn_now @ np.diag(lambda_n_now * (lambda_n_now > 0)) @ invRn_now)
        Am.append(Rn_now @ np.diag(lambda_n_now * (lambda_n_now < 0)) @ invRn_now)

    return An, absAn, Ap, Am

def get_local_face_indices(i_face, p):
    if i_face == 0:      # bottom (s = -1)
        return np.arange(0, p+1)
    elif i_face == 1:    # right  (r = +1)
        return np.arange(1, p+2) * (p + 1) - 1
    elif i_face == 2:    # top    (s = +1)
        return p * (p+1) + np.arange(0, p+1)
    elif i_face == 3:    # left   (r = -1)
        return np.arange(0, p+1) * (p + 1)
    else:
        raise ValueError("Face index must be 0..3")


def compute_flux_nodal(Q, i_element, j_element, n_elements, p, An, absAn, Ap, Am, gll_weights, dx, dy):
    """
    Q : shape (n_elements_tot, n_nodes_element_tot, 3)
        State vector at current time level.
    weights : 1D array of length p+1 (GLL weights).
    """

    index_element = func.get_element_index(i_element, j_element, n_elements)

    # local flux contribution: shape (n_nodes_element_tot, 3)
    flux = np.zeros_like(Q[index_element])

    # loop over the 4 faces of this element
    for i_face in range(4):

        An_now    = An[i_face]
        absAn_now = absAn[i_face]
        Ap_now    = Ap[i_face]
        Am_now    = Am[i_face]
        node_indices = get_local_face_indices(i_face, p)

        # Decide neighbor indices and edge quadrature weights
        if i_face == 0:  # bottom: neighbor at (i, j-1)
            ws = gll_weights * dx / 2.0
            has_neighbor = (j_element > 0)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element, j_element - 1, n_elements)
                node_indices_next = get_local_face_indices(2, p)

        elif i_face == 1:  # right: neighbor at (i+1, j)
            ws = gll_weights * dy / 2.0
            has_neighbor = (i_element < n_elements[0] - 1)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element + 1, j_element, n_elements)
                node_indices_next = get_local_face_indices(3, p)

        elif i_face == 2:  # top: neighbor at (i, j+1)
            ws = gll_weights * dx / 2.0
            has_neighbor = (j_element < n_elements[1] - 1)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element, j_element + 1, n_elements)
                node_indices_next = get_local_face_indices(0, p)

        elif i_face == 3:  # left: neighbor at (i-1, j)
            ws = gll_weights * dy / 2.0
            has_neighbor = (i_element > 0)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element - 1, j_element, n_elements)
                node_indices_next = get_local_face_indices(1, p)

        # loop over nodes on this face
        for i_node in range(p+1):

            U_minus = Q[index_element, node_indices[i_node], :]
            if has_neighbor:
                U_plus = Q[index_element_next, node_indices_next[i_node], :]
            else:
                # Mimicks homogeneous Neumann boundary conditions
                U_plus = np.zeros_like(U_minus)
                if i_face == 0 or i_face == 2:
                    # horizontal boundary: sigma_y = 0; sigma_x and v symmetric
                    U_plus[0] = U_minus[0]
                    U_plus[1] = U_minus[1]
                    U_plus[2] = -U_minus[2]
                elif i_face == 1 or i_face == 3:
                    # vertical boundary: sigma_x = 0; sigma_y and v symmetric
                    U_plus[0] = U_minus[0]
                    U_plus[1] = -U_minus[1]
                    U_plus[2] = U_minus[2]


            # Upwind numerical flux in normal direction
            # both versions should be equivalent
            # Fn_hat = Ap_now @ U_minus + Am_now @ U_plus
            Fn_hat = 0.5 * (An_now @ (U_minus + U_plus) + absAn_now @ (U_plus - U_minus))

            # add contribution to this DOF
            flux[node_indices[i_node], :] += ws[i_node] * Fn_hat

    return flux


def compute_flux_modal(Q, i_element, j_element, n_elements, p, An, absAn, Ap, Am, int_weights, dx, dy, shapes_diff0):
    """
    Q : shape (n_elements_tot, n_nodes_element_tot, 3)
        State vector at current time level.
    weights : 1D array of length p+1 (GLL weights).
    """

    index_element = func.get_element_index(i_element, j_element, n_elements)

    # local flux contribution: shape (n_nodes_element_tot, 3)
    flux = np.zeros_like(Q[index_element])

    # loop over the 4 faces of this element
    for i_face in range(4):

        An_now    = An[i_face]
        absAn_now = absAn[i_face]
        Ap_now    = Ap[i_face]
        Am_now    = Am[i_face]
        node_indices = get_local_face_indices(i_face, p)

        # Decide neighbor indices and edge quadrature weights
        if i_face == 0:  # bottom: neighbor at (i, j-1)
            ws = int_weights * dx / 2.0
            has_neighbor = (j_element > 0)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element, j_element - 1, n_elements)
                node_indices_next = get_local_face_indices(2, p)

        elif i_face == 1:  # right: neighbor at (i+1, j)
            ws = int_weights * dy / 2.0
            has_neighbor = (i_element < n_elements[0] - 1)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element + 1, j_element, n_elements)
                node_indices_next = get_local_face_indices(3, p)

        elif i_face == 2:  # top: neighbor at (i, j+1)
            ws = int_weights * dx / 2.0
            has_neighbor = (j_element < n_elements[1] - 1)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element, j_element + 1, n_elements)
                node_indices_next = get_local_face_indices(0, p)

        elif i_face == 3:  # left: neighbor at (i-1, j)
            ws = int_weights * dy / 2.0
            has_neighbor = (i_element > 0)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element - 1, j_element, n_elements)
                node_indices_next = get_local_face_indices(1, p)

        # loop integration points on this face
        for igauss, w in enumerate(ws):
            U_minus = np.zeros(3)
            for ifield in range(3):
                U_minus[ifield] = np.inner(shapes_diff0[igauss], Q[index_element, node_indices, ifield])

            U_plus = np.zeros_like(U_minus)
            if has_neighbor:
                for ifield in range(3):
                    U_plus[ifield] = np.inner(shapes_diff0[igauss], Q[index_element_next, node_indices_next, ifield])
            else:
                # Mimicks homogeneous Neumann boundary conditions
                if i_face == 0 or i_face == 2:
                    # horizontal boundary: sigma_y = 0; sigma_x and v symmetric
                    U_plus[0] = U_minus[0]
                    U_plus[1] = U_minus[1]
                    U_plus[2] = -U_minus[2]
                elif i_face == 1 or i_face == 3:
                    # vertical boundary: sigma_x = 0; sigma_y and v symmetric
                    U_plus[0] = U_minus[0]
                    U_plus[1] = -U_minus[1]
                    U_plus[2] = U_minus[2]

            # Upwind numerical flux in normal direction
            # both versions should be equivalent
            # Fn_hat = Ap_now @ U_minus + Am_now @ U_plus
            Fn_hat = 0.5 * (An_now @ (U_minus + U_plus) + absAn_now @ (U_plus - U_minus))

            # add contribution to this DOF
            for ifield in range(3):
                flux[node_indices, ifield] += w * shapes_diff0[igauss] * Fn_hat[ifield]

    return flux


def compute_rhs(dg_type, Q_stage, t, n_elements, p, An, absAn, Ap, Am, int_weights, dx, dy, Minv_all, Kx_all, Ky_all, F_all, ft, rho, c, shapes_diff0):
    """
    Q_stage : shape (n_elements_tot, n_nodes_element_tot, 3)
    returns dQdt with same shape
    """
    dQdt = np.zeros_like(Q_stage)

    for i_element in range(n_elements[0]):
        for j_element in range(n_elements[1]):
            index_element = func.get_element_index(i_element, j_element, n_elements)

            # flux for this element based on Q_stage
            if dg_type == "nodal":
                Flux = compute_flux_nodal(
                    Q_stage,
                    i_element, j_element,
                    n_elements, p,
                    An, absAn,
                    Ap, Am,
                    int_weights,
                    dx, dy
                )
            elif dg_type == "modal":
                Flux = compute_flux_modal(
                    Q_stage,
                    i_element, j_element,
                    n_elements, p,
                    An, absAn,
                    Ap, Am,
                    int_weights,
                    dx, dy,
                    shapes_diff0,
                )
            else:
                raise NotImplementedError

            Me_inv = Minv_all[index_element]
            Kx     = Kx_all[index_element]
            Ky     = Ky_all[index_element]
            F      = F_all[index_element]

            v  = Q_stage[index_element, :, 0]
            sx = Q_stage[index_element, :, 1]
            sy = Q_stage[index_element, :, 2]

            # same formulas you currently use, just written as rhs
            rhs_v  = Me_inv * (
                F * ft(t)
                - (1.0 / rho) * (Kx @ sx)
                - (1.0 / rho) * (Ky @ sy)
                + Flux[:, 0]
            )

            rhs_sx = Me_inv * (
                - rho * c * c * (Kx @ v)
                + Flux[:, 1]
            )

            rhs_sy = Me_inv * (
                - rho * c * c * (Ky @ v)
                + Flux[:, 2]
            )

            dQdt[index_element, :, 0] = rhs_v
            dQdt[index_element, :, 1] = rhs_sx
            dQdt[index_element, :, 2] = rhs_sy

    return dQdt



def compute_rhs_fcm(Q_stage, t, n_elements, p, An, absAn, Minv_all, Kx_all, Ky_all, F_all, ft, rho, c, shapes, weights_all, points_all):
    """
    Q_stage : shape (n_elements_tot, n_nodes_element_tot, 3)
    returns dQdt with same shape
    """
    dQdt = np.zeros_like(Q_stage)

    for j_element in range(n_elements[1]):
        for i_element in range(n_elements[0]):
            index_element = func.get_element_index(i_element, j_element, n_elements)

            Flux = compute_flux_modal_fcm(
                Q_stage,
                i_element, j_element,
                n_elements, p,
                An, absAn,
                shapes,
                weights_all, points_all
            )

            Me_inv = Minv_all[index_element]
            Kx     = Kx_all[index_element]
            Ky     = Ky_all[index_element]
            F      = F_all[index_element]

            v  = Q_stage[index_element, :, 0]
            sx = Q_stage[index_element, :, 1]
            sy = Q_stage[index_element, :, 2]

            # same formulas you currently use, just written as rhs
            rhs_v  = Me_inv @ (
                F * ft(t)
                - (1.0 / rho) * (Kx @ sx)
                - (1.0 / rho) * (Ky @ sy)
                + Flux[:, 0]
            )

            rhs_sx = Me_inv @ (
                - rho * c * c * (Kx @ v)
                + Flux[:, 1]
            )

            rhs_sy = Me_inv @ (
                - rho * c * c * (Ky @ v)
                + Flux[:, 2]
            )

            dQdt[index_element, :, 0] = rhs_v
            dQdt[index_element, :, 1] = rhs_sx
            dQdt[index_element, :, 2] = rhs_sy

    return dQdt

def compute_flux_modal_fcm(Q, i_element, j_element, n_elements, p, An, absAn, shapes, weights_all, points_all):
    """
    Q : shape (n_elements_tot, n_nodes_element_tot, 3)
        State vector at current time level.
    weights : 1D array of length p+1 (GLL weights).
    """

    index_element = func.get_element_index(i_element, j_element, n_elements)
    weights_now = weights_all[index_element]
    points_now = points_all[index_element]

    # local flux contribution: shape (n_nodes_element_tot, 3)
    flux = np.zeros_like(Q[index_element])

    # loop over the 4 faces of this element
    for i_face in range(4):

        An_now    = An[i_face]
        absAn_now = absAn[i_face]
        node_indices = get_local_face_indices(i_face, p)

        # Decide neighbor indices and edge quadrature weights
        if i_face == 0:  # bottom: neighbor at (i, j-1)
            has_neighbor = (j_element > 0)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element, j_element - 1, n_elements)
                node_indices_next = get_local_face_indices(2, p)

        elif i_face == 1:  # right: neighbor at (i+1, j)
            has_neighbor = (i_element < n_elements[0] - 1)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element + 1, j_element, n_elements)
                node_indices_next = get_local_face_indices(3, p)

        elif i_face == 2:  # top: neighbor at (i, j+1)
            has_neighbor = (j_element < n_elements[1] - 1)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element, j_element + 1, n_elements)
                node_indices_next = get_local_face_indices(0, p)

        elif i_face == 3:  # left: neighbor at (i-1, j)
            has_neighbor = (i_element > 0)
            if has_neighbor:
                index_element_next = func.get_element_index(i_element - 1, j_element, n_elements)
                node_indices_next = get_local_face_indices(1, p)

        ws = weights_now[i_face]
        points = points_now[i_face]
        shapes_diff0 = [np.array([shape(r) for shape in shapes]) for r in points]

        # loop integration points on this face
        for igauss, w in enumerate(ws):
            U_minus = np.zeros(3)
            for ifield in range(3):
                U_minus[ifield] = np.inner(shapes_diff0[igauss], Q[index_element, node_indices, ifield])

            U_plus = np.zeros_like(U_minus)
            if has_neighbor:
                for ifield in range(3):
                    U_plus[ifield] = np.inner(shapes_diff0[igauss], Q[index_element_next, node_indices_next, ifield])
            else:
                # Mimicks homogeneous Neumann boundary conditions
                if i_face == 0 or i_face == 2:
                    # horizontal boundary: sigma_y = 0; sigma_x and v symmetric
                    U_plus[0] = U_minus[0]
                    U_plus[1] = U_minus[1]
                    U_plus[2] = -U_minus[2]
                elif i_face == 1 or i_face == 3:
                    # vertical boundary: sigma_x = 0; sigma_y and v symmetric
                    U_plus[0] = U_minus[0]
                    U_plus[1] = -U_minus[1]
                    U_plus[2] = U_minus[2]

            # Upwind numerical flux in normal direction
            # both versions should be equivalent
            # Fn_hat = Ap_now @ U_minus + Am_now @ U_plus
            Fn_hat = 0.5 * (An_now @ (U_minus + U_plus) + absAn_now @ (U_plus - U_minus))

            # add contribution to this DOF
            for ifield in range(3):
                flux[node_indices, ifield] += w * shapes_diff0[igauss] * Fn_hat[ifield]

    return flux