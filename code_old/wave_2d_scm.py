import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

import utils_sem as sem
import fcm
import func


# Domain definition
lengths = [2, 2]
duration = 2.0

# definition of the domain
# center = [2.0, 0.0]
# radius = 0.75
# domain = lambda x, y : (x - center[0])**2 + (y - center[1])**2 >= radius**2
end_of_domain_x = 1.5
domain = lambda x, y : x <= end_of_domain_x

# material
alpha = lambda x, y : 1.0 if domain( x, y ) else .001
rho = lambda x, y : alpha( x, y )
c = lambda x, y : 1

# Source function (Ricklers wavelet)
frequency = 10
t0 = 1.0 / frequency
sigma_time = 1.0 / (2.0 * np.pi * frequency)
wavelength = 1.0 / frequency
# spatial distribution
center_source = [.75, 1.0]
sigma_space = wavelength / 4

ft = lambda t: -(t - t0) / (np.sqrt(2 * np.pi) * sigma_time ** 3) * np.exp(-(t - t0) ** 2 / (2 * sigma_time ** 2))
fx = lambda x, y: 10 * np.exp(-((x - center_source[0]) ** 2 + (y - center_source[1]) ** 2) / (2 * sigma_space ** 2))
# fx = lambda x, y: 10 * np.exp(-((x - center_source[0]) ** 2) / (2 * sigma_space ** 2))

# Discretization
n_elements = [20, 20]
n_steps = 400
p = 4
# integration
quadrature_order = p + 1
depth = p + 1
n_seed = p + 2

dt = duration / n_steps
dx = lengths[0] / n_elements[0]
dy = lengths[1] / n_elements[1]

# Prepare Lagrange polynomials and integration
gl_coordinates, gl_weights = np.polynomial.legendre.leggauss(quadrature_order)
gll_coordinates, gll_weights = func.GLL(p + 1)

# create shape functions
lagrange_coords = gll_coordinates
lagrange_values = np.identity(p + 1)
lagrange = lambda i: scipy.interpolate.lagrange(lagrange_coords, lagrange_values[i])

shapes_diff0 = [lagrange(i) for i in range(p + 1)]
shapes_diff1 = [np.polyder(shape) for shape in shapes_diff0]

n_dof_element = (p + 1) ** 2
n_dof_direction = [n_elements[0] * p + 1, n_elements[1] * p + 1]
n_dof_global = n_dof_direction[0] * n_dof_direction[1]

print("Assembling (" + str(n_dof_global) + " dofs, " + str(n_elements[0] * n_elements[1]) + " elements) ... ", flush=True)

# Allocate data structure for coordinate format
row = np.zeros(n_dof_element ** 2 * n_elements[0] * n_elements[1], dtype=np.uint)
col = np.zeros(n_dof_element ** 2 * n_elements[0] * n_elements[1], dtype=np.uint)
Kdata = np.zeros(n_dof_element ** 2 * n_elements[0] * n_elements[1])
Mdata = np.zeros(n_dof_element ** 2 * n_elements[0] * n_elements[1])
Fx = np.zeros(n_dof_global)

mapX = lambda ielement, r: (ielement + r / 2.0 + 0.5) * dx
mapY = lambda jelement, s: (jelement + s / 2.0 + 0.5) * dy

# Assemble lumped mass matrix
for j_element in range(n_elements[1]):
    for i_element in range(n_elements[0]):
        Me = np.zeros((n_dof_element, n_dof_element))
        Ke = np.zeros((n_dof_element, n_dof_element))
        Fe = np.zeros((n_dof_element,))

        location_map = sem.create_location_map(i_element, j_element, n_dof_direction, p)
        partitions = fcm.generate_quadtree(i_element, j_element, depth, n_seed, mapX, mapY, domain)
        cell_is_cut = len(partitions) > 1

        for (r0, r1), (s0, s1) in partitions:

            if cell_is_cut:
                rValues = [xi * (r1 - r0) / 2 + (r1 + r0) / 2 for xi in gl_coordinates]
                sValues = [et * (s1 - s0) / 2 + (s1 + s0) / 2 for et in gl_coordinates]
                weights_now = gl_weights
            else:
                rValues = [xi * (r1 - r0) / 2 + (r1 + r0) / 2 for xi in gll_coordinates]
                sValues = [et * (s1 - s0) / 2 + (s1 + s0) / 2 for et in gll_coordinates]
                weights_now = gll_weights

            # Evaluate Lagrange polynomials in both directions
            shapes_r_diff0 = [np.array([shape(r) for shape in shapes_diff0]) for r in rValues]
            shapes_s_diff0 = [np.array([shape(s) for shape in shapes_diff0]) for s in sValues]

            for igauss, (r, w0) in enumerate(zip(rValues, weights_now)):
                for jgauss, (s, w1) in enumerate(zip(sValues, weights_now)):
                    x, y = mapX(i_element, r), mapY(j_element, s)
                    weight = w0 * w1 * dx / 2 * (r1 - r0) / 2 * dy / 2 * (s1 - s0) / 2

                    # Compute tensor product and derivatives w.r.t. x and y
                    N = np.outer(shapes_s_diff0[jgauss], shapes_r_diff0[igauss]).ravel()

                    # Add contributions
                    Me += np.outer(N, N) * rho(x, y) * weight

            # Map from integration cell to element
            rValues = [xi * (r1 - r0) / 2 + (r1 + r0) / 2 for xi in gl_coordinates]
            sValues = [et * (s1 - s0) / 2 + (s1 + s0) / 2 for et in gl_coordinates]
            weights_now = gl_weights

            # Evaluate Lagrange polynomials in both directions
            shapes_r_diff0 = [np.array([shape(r) for shape in shapes_diff0]) for r in rValues]
            shapes_r_diff1 = [np.array([shape(r) for shape in shapes_diff1]) for r in rValues]
            shapes_s_diff0 = [np.array([shape(s) for shape in shapes_diff0]) for s in sValues]
            shapes_s_diff1 = [np.array([shape(s) for shape in shapes_diff1]) for s in sValues]

            for igauss, (r, w0) in enumerate(zip(rValues, weights_now)):
                for jgauss, (s, w1) in enumerate(zip(sValues, weights_now)):
                    x, y = mapX(i_element, r), mapY(j_element, s)
                    weight = w0 * w1 * dx / 2 * (r1 - r0) / 2 * dy / 2 * (s1 - s0) / 2

                    # Compute tensor product and derivatives w.r.t. x and y
                    N = np.outer(shapes_s_diff0[jgauss], shapes_r_diff0[igauss]).ravel()
                    dNdx = np.outer(shapes_s_diff0[jgauss], shapes_r_diff1[igauss] * 2/dx).ravel()
                    dNdy = np.outer(shapes_s_diff1[jgauss] * 2/dy, shapes_r_diff0[igauss]).ravel()

                    # Add contributions
                    Ke += (np.outer(dNdx, dNdx) + np.outer(dNdy, dNdy)) * rho(x, y) * c(x, y) * c(x, y) * weight
                    Fe += N * fx(x, y) * weight


        # Repeat locationMaps as colums/rows linearize the result to obtain matrix entry coordinates
        element_slice = sem.create_element_slice(i_element, j_element, n_elements, n_dof_element)

        row[element_slice] = np.broadcast_to(location_map, (n_dof_element, n_dof_element)).T.ravel()
        col[element_slice] = np.broadcast_to(location_map, (n_dof_element, n_dof_element)).ravel()

        Mdata[element_slice] = Me.ravel()
        Kdata[element_slice] = Ke.ravel()
        Fx[location_map] += Fe

# Create coordinate matrix and convert to compressed sparse column format
M = scipy.sparse.coo_matrix((Mdata, (row, col)), shape=(n_dof_global, n_dof_global)).tocsc()
K = scipy.sparse.coo_matrix((Kdata, (row, col)), shape=(n_dof_global, n_dof_global)).tocsc()

print("Average bandwidth before filtering: " + "%.2f" % (M.nnz / M.shape[0]))
M.data[np.abs(M.data) < 1e-16 * scipy.sparse.linalg.norm(M)] = 0.0
M.eliminate_zeros()
print("Average bandwidth after filtering: " + "%.2f" % (M.nnz / M.shape[0]))

print("Time integration (" + str(n_steps) + " time steps) ... ", flush=True)

# Sparse LU factorization and time integration
factorized = scipy.sparse.linalg.splu(M)

u = np.zeros((n_steps + 1, n_dof_global))

for i in range(2, n_steps + 1):
    u[i] = factorized.solve(M * (2 * u[i - 1] - u[i - 2]) + dt ** 2 * (Fx * ft(i * dt) - K * u[i - 1]))

print("Plotting ... ", flush=True)

# Plot animation (at Lagrange interpolation points)
x = [mapX(ielement, r) for ielement in range(n_elements[0]) for r in lagrange_coords[:-1]]
y = [mapY(jelement, s) for jelement in range(n_elements[1]) for s in lagrange_coords[:-1]]

x += [mapX(n_elements[0] - 1, lagrange_coords[-1])]
y += [mapY(n_elements[1] - 1, lagrange_coords[-1])]

X, Y = np.meshgrid(x, y, indexing='ij')

Xe, Ye = np.meshgrid(np.linspace(0, lengths[0], n_elements[0] + 1),
                     np.linspace(0, lengths[1], n_elements[1] + 1), indexing='ij')

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')


def animate(i):
    ax.clear()
    Z = np.reshape(u[i], (n_dof_direction[1], n_dof_direction[0])).T
    ax.contourf(X, Y, Z, levels=np.linspace(-1.0, 1.0, 24), cmap='seismic', extend='both')

    plt.plot(Xe, Ye, 'black')
    plt.plot(Xe.T, Ye.T, 'black')


anim = animation.FuncAnimation(fig, animate, range(0, n_steps, 4), interval=1)

plt.show()

# Plot last state
Z = np.reshape(u[-1], (n_dof_direction[1], n_dof_direction[0])).T

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.contourf(X, Y, Z, levels=np.linspace(-1.0, 1.0, 24), cmap='seismic', extend='both')
ax.plot(Xe, Ye, 'black')
ax.plot(Xe.T, Ye.T, 'black')
plt.show()
