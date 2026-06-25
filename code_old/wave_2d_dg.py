import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

import utils_dg as dg
import func

# Choose mode of excitation: either with external force or with initial condition
excitation_mode = "external" # "initial" or "external"

# Choose time integration: Either Euler or Runge Kutta
time_scheme = "RK2" # "Euler" or "RK2"

# Choose DG type: nodal or modal point of view
# (basically just the integration for F, Kx, Ky, and the flux changes from GLL (nodal) to GL (modal))
dg_type = "nodal" # "nodal" or "modal"

# Domain definition
lengths = [2, 2]
duration = 2.0

# material
rho = 1
c = 1.0

# normals of the elementa
normals = [np.array([0, -1]), np.array([1, 0]), np.array([0, 1]), np.array([-1, 0])]
# pre-computations for flux
An, absAn, Ap, Am = dg.compute_A_matrices(rho, c, normals)

# Discretization
n_elements = [10, 10]
n_elements_tot = n_elements[0] * n_elements[1]
p = 4
n_steps = 400

dt = duration / n_steps
dx = lengths[0] / n_elements[0]
dy = lengths[1] / n_elements[1]

# sizes
n_nodes_element = [p+1, p+1]
n_nodes_element_tot = n_nodes_element[0] * n_nodes_element[1]
n_nodes_global = [n_nodes_element[0]*n_elements[0], n_nodes_element[1]*n_elements[1]]
n_nodes_global_tot = n_nodes_global[0] * n_nodes_global[1]

# Prepare Lagrange polynomials
# GLL points for shape functions and integration of M
integration_coordinates_1, integration_weights_1 = func.GLL(p + 1)
lagrange_coords = integration_coordinates_1
lagrange_values = np.identity(p + 1)
lagrange = lambda i: scipy.interpolate.lagrange(lagrange_coords, lagrange_values[i])
shapes_diff0 = [lagrange(i) for i in range(p + 1)]
shapes_diff1 = [np.polyder(shape) for shape in shapes_diff0]

# Evaluation of functions at GLL points
shapes_r_diff0_1 = [np.array([shape(r) for shape in shapes_diff0]) for r in integration_coordinates_1]
shapes_s_diff0_1 = [np.array([shape(s) for shape in shapes_diff0]) for s in integration_coordinates_1]

# definition of second integration rule for F, Kx, Ky, and the flux
if dg_type == "nodal":
    # also GLL points
    integration_coordinates_2, integration_weights_2 = integration_coordinates_1, integration_weights_1
elif dg_type == "modal":
    # GL points!!
    integration_coordinates_2, integration_weights_2 = np.polynomial.legendre.leggauss(p + 1)
else:
    raise NotImplementedError

# evaluation of Lagrange polynomials and derivatives for the second integration rule
shapes_r_diff0_2 = [np.array([shape(r) for shape in shapes_diff0]) for r in integration_coordinates_2]
shapes_s_diff0_2 = [np.array([shape(s) for shape in shapes_diff0]) for s in integration_coordinates_2]
shapes_r_diff1_2 = [np.array([shape(r) for shape in shapes_diff1]) for r in integration_coordinates_2]
shapes_s_diff1_2 = [np.array([shape(s) for shape in shapes_diff1]) for s in integration_coordinates_2]

# Element mapping
mapX = lambda ielement, r: (ielement + r / 2.0 + 0.5) * dx
mapY = lambda jelement, s: (jelement + s / 2.0 + 0.5) * dy

# global node vector
xg = np.array([mapX(i, integration_coordinates_1) for i in range(n_elements[0])]).ravel()
yg = np.array([mapY(j, integration_coordinates_1) for j in range(n_elements[1])]).ravel()
# global node numbering in grid
Y, X = np.meshgrid(xg, yg, indexing='ij')
Ye, Xe = np.meshgrid(np.linspace(0, lengths[0], n_elements[0] + 1),
                     np.linspace(0, lengths[1], n_elements[1] + 1), indexing='ij')

if excitation_mode == "external":
    # External excitation - Source function (Ricker wavelet)
    frequency = 5
    t0 = 1.0 / frequency
    sigma_time = 1.0 / (2.0 * np.pi * frequency)
    # spatial distribution
    center_source = [1.0, 1.0]
    sigma_space = 0.06

    ft = lambda t: -(t - t0) / (np.sqrt(2 * np.pi) * sigma_time ** 3) * np.exp(-(t - t0) ** 2 / (2 * sigma_time ** 2))
    fx = lambda x, y: 10 * np.exp(-((x - center_source[0]) ** 2 + (y - center_source[1]) ** 2) / (2 * sigma_space ** 2))

    # zero initial condition
    v0 = np.zeros_like(X)
    sigma_x0 = np.zeros_like(X)
    sigma_y0 = np.zeros_like(X)

elif excitation_mode == "initial":
    # Initial condition - Gauss curve
    sig    = .2
    xy0 = [1.0, 1.0]
    v0 = np.exp(-1./sig**2 * ((X-xy0[0])**2 + (Y-xy0[1])**2))
    sigma_x0 = np.zeros_like(X)
    sigma_y0 = np.zeros_like(X)

    # no external excitation
    ft = lambda t: 0
    fx = lambda x, y: 0

    # Plot the initial condition
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
        ax.plot(Xe, Ye, 'black')
        ax.plot(Ye, Xe, 'black')
    axes[0].set_title(r"Velocity $v$")
    axes[0].contourf(X, Y, v0, levels=np.linspace(-.5, .5, 24), cmap='seismic', extend='both')
    axes[1].set_title(r"Stress $\sigma_x$")
    axes[1].contourf(X, Y, sigma_x0, levels=np.linspace(-.5, .5, 24), cmap='seismic', extend='both')
    axes[2].set_title(r"Stress $\sigma_y$")
    axes[2].contourf(X, Y, sigma_y0, levels=np.linspace(-.5, .5, 24), cmap='seismic', extend='both')

    plt.show()

else:
    raise NotImplementedError


# Pre-compute element matrices
Minv_all = []
Kx_all = []
Ky_all = []
F_all = []

# Order of for loop important. It has to match with the element numbering
for j_element in range(n_elements[1]):
    for i_element in range(n_elements[0]):
        # Allocate element matrices
        Me = np.zeros((n_nodes_element_tot, n_nodes_element_tot))
        Kex = np.zeros((n_nodes_element_tot, n_nodes_element_tot))
        Key = np.zeros((n_nodes_element_tot, n_nodes_element_tot))
        Fe = np.zeros((n_nodes_element_tot,))

        # Loop over integration points - first integration rule: gll integration
        for igauss, (r, w0) in enumerate(zip(integration_coordinates_1, integration_weights_1)):
            for jgauss, (s, w1) in enumerate(zip(integration_coordinates_1, integration_weights_1)):
                weight = w0 * dx / 2 * w1 * dy / 2

                # Compute tensor product and derivatives w.r.t. x and y
                N = np.outer(shapes_s_diff0_1[jgauss], shapes_r_diff0_1[igauss]).ravel()

                # Add contributions
                Me += weight * np.outer(N, N)


        # Loop over integration points - second integration rule (either GLL or FL)
        for igauss, (r, w0) in enumerate(zip(integration_coordinates_2, integration_weights_2)):
            for jgauss, (s, w1) in enumerate(zip(integration_coordinates_2, integration_weights_2)):
                # mapping for source evaluation
                x, y = mapX(i_element, r), mapY(j_element, s)
                weight = w0 * dx / 2 * w1 * dy / 2

                # Compute tensor product and derivatives w.r.t. x and y
                N = np.outer(shapes_s_diff0_2[jgauss], shapes_r_diff0_2[igauss]).ravel()
                dNdx = np.outer(shapes_s_diff0_2[jgauss], shapes_r_diff1_2[igauss] * 2 / dx).ravel()
                dNdy = np.outer(shapes_s_diff1_2[jgauss] * 2 / dy, shapes_r_diff0_2[igauss]).ravel()

                Kex += weight * np.outer(dNdx, N)
                Key += weight * np.outer(dNdy, N)
                Fe += 1/rho * weight * N * fx(x, y)

        # Save all element matrices (Me is directly inverted and just the diagonal is saved)
        Minv_all.append(1 / np.diag(Me))
        Kx_all.append(Kex)
        Ky_all.append(Key)
        F_all.append(Fe)

# Allocate Solution vector; displacement u is integrated along the time integraiton
# Q size: time steps x elements x nodes x fields (v, sigmax, sigmay)
U = np.zeros((n_steps+1, n_elements_tot, n_nodes_element_tot))
Q = np.zeros((n_steps+1, n_elements_tot, n_nodes_element_tot, 3))

# Set initial conditions in Q
v0 = v0.ravel()
sigma_x0 = sigma_x0.ravel()
sigma_y0 = sigma_y0.ravel()
for i_element in range(n_elements[0]):
    for j_element in range(n_elements[1]):
        index_element = func.get_element_index(i_element, j_element, n_elements)
        lm = dg.create_location_map(i_element, j_element, n_nodes_global, p)
        Q[0, index_element, :, 0] = v0[lm]
        Q[0, index_element, :, 1] = sigma_x0[lm]
        Q[0, index_element, :, 2] = sigma_y0[lm]

# Time integration
for n in range(n_steps):
    t_n = n * dt

    # stage 1: Euler step
    k1 = dg.compute_rhs(dg_type, Q[n], t_n, n_elements, p, An, absAn, Ap, Am, integration_weights_2, dx, dy, Minv_all, Kx_all, Ky_all, F_all, ft, rho, c, shapes_r_diff0_2)
    k1_U = Q[n,:,:,0]

    if time_scheme == "Euler":
        Q[n + 1] = Q[n] + dt * k1
        U[n + 1] = U[n] + dt * k1_U

    elif time_scheme == "RK2":
        Q1 = Q[n] + dt * k1
        # stage 2:
        k2 = dg.compute_rhs(dg_type, Q1, t_n + dt, n_elements, p, An, absAn, Ap, Am, integration_weights_2, dx, dy, Minv_all, Kx_all, Ky_all, F_all, ft, rho, c, shapes_r_diff0_2)
        k2_U = Q1[:,:,0]
        # SSPRK2 update:
        Q[n+1] = Q[n] + 0.5 * dt * (k1 + k2)
        U[n+1] = U[n] + 0.5 * dt * (k1_U + k2_U)

    else:
        raise NotImplementedError

    # Track maximum velocity to see if integration is stable
    print(f"Time step {n} from {n_steps}: Maximum velocity {np.amax(Q[n+1, :, :, 0])}")


print("Plotting ... ", flush=True)

fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax in axes:
    ax.set_aspect('equal', adjustable='box')

u_plot = np.zeros(n_nodes_global_tot)
v_plot = np.zeros(n_nodes_global_tot)
sigma_x_plot = np.zeros(n_nodes_global_tot)
sigma_y_plot = np.zeros(n_nodes_global_tot)

abs_u_max = np.amax(U)
abs_v_max = np.amax(Q[:,:,:,0])
abs_sx_max = np.amax(Q[:,:,:,1])
abs_sy_max = np.amax(Q[:,:,:,2])

def animate(i):
    for ax in axes:
        ax.clear()
        ax.plot(Xe, Ye, 'black')
        ax.plot(Ye, Xe, 'black')

    for i_element in range(n_elements[0]):
        for j_element in range(n_elements[1]):
            index_element = func.get_element_index(i_element, j_element, n_elements)
            lm = dg.create_location_map(i_element, j_element, n_nodes_global, p)
            u_plot[lm] = U[i, index_element, :]
            v_plot[lm] = Q[i, index_element, :, 0]
            sigma_x_plot[lm] = Q[i, index_element, :, 1]
            sigma_y_plot[lm] = Q[i, index_element, :, 2]

    u_grid = np.reshape(u_plot, (n_nodes_global[0], n_nodes_global[1]))
    v_grid = np.reshape(v_plot, (n_nodes_global[0], n_nodes_global[1]))
    sigma_x_grid = np.reshape(sigma_x_plot, (n_nodes_global[0], n_nodes_global[1]))
    sigma_y_grid = np.reshape(sigma_y_plot, (n_nodes_global[0], n_nodes_global[1]))

    axes[0].set_title(r"Displacement $u$")
    axes[0].contourf(X, Y, u_grid, levels=np.linspace(-abs_u_max/2, abs_u_max/2, 24), cmap='seismic', extend='both')
    axes[1].set_title(r"Velocity $v$")
    axes[1].contourf(X, Y, v_grid, levels=np.linspace(-abs_v_max/2, abs_v_max/2, 24), cmap='seismic', extend='both')
    axes[2].set_title(r"Stress $\sigma_x$")
    axes[2].contourf(X, Y, sigma_x_grid, levels=np.linspace(-abs_sx_max/2, abs_sx_max/2, 24), cmap='seismic', extend='both')
    axes[3].set_title(r"Stress $\sigma_y$")
    axes[3].contourf(X, Y, sigma_y_grid, levels=np.linspace(-abs_sy_max/2, abs_sy_max/2, 24), cmap='seismic', extend='both')

anim = animation.FuncAnimation(fig, animate, range(0, n_steps, 4), interval=1)

plt.show()

# Plot last step
for i_element in range(n_elements[0]):
    for j_element in range(n_elements[1]):
        index_element = func.get_element_index(i_element, j_element, n_elements)
        lm = dg.create_location_map(i_element, j_element, n_nodes_global, p)
        u_plot[lm] = U[-1, index_element, :]
u_grid = np.reshape(u_plot, (n_nodes_global[0], n_nodes_global[1]))

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.contourf(X, Y, u_grid, levels=np.linspace(-1.0, 1.0, 24), cmap='seismic', extend='both')
ax.plot(Xe, Ye, 'black')
ax.plot(Ye, Xe, 'black')
plt.show()
