import numpy as np
from typing import Callable, Optional, Tuple

def generate_quadtree(i_element: int,
                      j_element: int,
                      max_depth: int,
                      n_seed: int,
                      mapX: Callable,
                      mapY: Callable,
                      io_test,):

    begin, end, depth, partitions = 0, 1, 0, []
    tree = [((-1.0, 1.0), (-1.0, 1.0))]

    while begin != end and depth < max_depth:
        for (r0, r1), (s0, s1) in tree[begin:end]:

            # Create grid of points and evaluate embedded domain
            seedX = np.linspace(mapX(i_element, r0), mapX(i_element, r1), n_seed)
            seedY = np.linspace(mapY(j_element, s0), mapY(j_element, s1), n_seed)

            X, Y = np.meshgrid(seedX, seedY, indexing='ij')

            result = io_test(X.ravel(), Y.ravel())

            # Subdivide if some are inside and some outside
            if not np.all(result) and not np.all(np.logical_not(result)):
                tree += [((r0, (r0 + r1) / 2), (s0, (s0 + s1) / 2)),
                         (((r0 + r1) / 2, r1), (s0, (s0 + s1) / 2)),
                         ((r0, (r0 + r1) / 2), ((s0 + s1) / 2, s1)),
                         (((r0 + r1) / 2, r1), ((s0 + s1) / 2, s1))]
            else:
                partitions.append(((r0, r1), (s0, s1)))

        begin, end, depth = end, len(tree), depth + 1

    return partitions + tree[begin:end]

def face_bisection(io_test: Callable[[float, float], bool],
                   a: np.ndarray,
                   b: np.ndarray,
                   max_iter: int = 20,):

    # Bisection on t in [0,1]
    t1, t2 = 0.0, 1.0
    io1 = io_test(a[0], a[1])
    io2 = io_test(b[0], b[1])

    for _ in range(max_iter):
        tM = 0.5 * (t1 + t2)
        pM = a + tM * (b - a)
        ioM = io_test(pM[0], pM[1])

        if io1 == ioM:
            t1 = tM
        elif io2 == ioM:
            t2 = tM
        else:
            raise Exception("Check if face is really just cut once")

    return pM

def get_face_nodes(i_element, j_element, i_face, dx, dy):
    # 0 - bottom, 1 - right, 2 - top, 3 - left
    if i_face == 0:
        a = np.array([i_element * dx, j_element * dy])
        b = np.array([(i_element + 1) * dx, j_element * dy])
    elif i_face == 1:
        a = np.array([(i_element + 1) * dx, j_element * dy])
        b = np.array([(i_element + 1) * dx, (j_element + 1) * dy])
    elif i_face == 2:
        a = np.array([i_element * dx, (j_element + 1) * dy])
        b = np.array([(i_element + 1) * dx, (j_element + 1) * dy])
    elif i_face == 3:
        a = np.array([i_element * dx, j_element * dy])
        b = np.array([i_element * dx, (j_element + 1) * dy])
    else:
        raise Exception("Quadrilateral just has four faces")

    return a, b

def distribute_on_face(i_element, j_element, domain, i_f, integration_weights, integration_coordinates, dx, dy, alpha0, inv_mapX, inv_mapY):

    if i_f == 0 or i_f == 2:
        width = dx
    elif i_f == 1 or i_f == 3:
        width = dy

    a, b = get_face_nodes(i_element, j_element, i_f, dx, dy)

    if (domain(a[0], a[1]) == True) and (domain(b[0], b[1]) == True):
        # the face is not cut and inside
        integration_weights_now = integration_weights * width / 2.0
        integration_points_now = integration_coordinates
    elif (domain(a[0], a[1]) == False) and (domain(b[0], b[1]) == False):
        # the face is not cut and outside
        integration_weights_now = alpha0 * integration_weights * width / 2.0
        integration_points_now = integration_coordinates
    else:
        # the face is cut
        # determine cut position in global and local COS
        c = face_bisection(domain, a, b)
        if i_f == 0 or i_f == 2:
            c_local = inv_mapX(i_element, c[0])
        elif i_f == 1 or i_f == 3:
            c_local = inv_mapY(j_element, c[1])

        if domain(a[0], a[1]) == True:
            integration_weights_now = np.hstack((integration_weights * width / 2.0 * (c_local + 1) / 2.0,
                                                 alpha0 * integration_weights * width / 2.0 * (1 - c_local) / 2.0))
        else:
            integration_weights_now = np.hstack((alpha0 * integration_weights * dx / 2.0 * (c_local + 1) / 2.0,
                                                 integration_weights * dx / 2.0 * (1 - c_local) / 2.0))
        integration_points_now= np.hstack((integration_coordinates * (c_local + 1) / 2 + (c_local - 1) / 2,
                                           integration_coordinates * (1 - c_local) / 2 + (1 + c_local) / 2))

    return integration_points_now, integration_weights_now