import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
from tqdm import tqdm

import material
import mesh_dg as mesh
import integration
import ansatz
import functions
import geometries
import spacetree
import fluxes


# Spatial dimension
dim = 1
print("Setting up 1D DG problem...")

# computational domain
origin = np.array([0.0])
length = np.array([1.0])
print(f"Domain origin: {origin}, length: {length}")

# mesh
n_elements = np.array([20])
polynomial_degree = np.array([4])
print(f"Number of elements: {n_elements}, Polynomial degree: {polynomial_degree}")
mesh_1d = mesh.ndRectangle(dim=dim, origin=origin, lengths=length, n_elements=n_elements, polynomial_degree=polynomial_degree)

# physical domain (immersed geometry)
x_bounds = [0.0, .82]
domain = geometries.immersed_1d_bar(x_bounds=x_bounds)
# domain = geometries.all_physical()

# stabilization parameters for cut cells and parameter for spacetree
alpha = 1e-2
depth = np.max(polynomial_degree) + 1
n_seed = 10

# material 
density = 1.0
wave_speed = 1.0
print(f"Material properties - Density: {density}, Wave Speed: {wave_speed}")
mat = material.acousticMaterial(density=density, wave_speed=wave_speed)

# normals of the elements and pre-computations for fluxes
normals = [np.array([-1,]), np.array([1,])]
An, absAn, An_fict, absAn_fict = fluxes.compute_A_matrices(density, wave_speed, alpha, normals)

# time
T = x_bounds[1] / wave_speed
n_steps = 20000
print(f"Total simulation time: {T}, Number of time steps: {n_steps}")
dt = T / n_steps
time_scheme = "Euler" # "Euler" or "RK2"

# initial conditions
frequency = 1.0
x0 = x_bounds[1] / 2
sigma = wave_speed / frequency / 20
print(f"Initial condition - Gaussian pulse centered at {x0} with sigma {sigma}")
ic_func = functions.gaussian_pulse_1d(x0=x0, sigma=sigma, wave_speed=wave_speed)
def ic_0_func_v(x):
    return ic_func(x, 0.0)
def ic_0_func_sig(x):
    return 0.0

# external force
def f_ex_func_v(x): 
    return 0.0
def f_ex_func_sig(x): 
    return 0.0
def f_t(t):
    return 0.0

# defining shape functions
gll_nodes = [integration.get_gll_integration(polynomial_degree[i_d] + 1)[0] for i_d in range(dim)]
shapes_0 = [ansatz.get_lagrange_functions(gll_nodes[0], derivative_order=0)]
shapes_1 = [ansatz.get_lagrange_functions(gll_nodes[0], derivative_order=1)]

# integration points and weights
int_points_rule1, int_weights_rule1 = [], []
int_points_rule2, int_weights_rule2 = [], []
for i_d in range(dim):
    # int_p, int_w = integration.get_gll_integration(polynomial_degree[i_d] + 1)
    int_p, int_w = np.polynomial.legendre.leggauss(polynomial_degree[i_d] + 1)
    int_points_rule1.append(int_p)
    int_weights_rule1.append(int_w)
    int_p, int_w = np.polynomial.legendre.leggauss(polynomial_degree[i_d] + 1)
    int_points_rule2.append(int_p)
    int_weights_rule2.append(int_w)

print("1D DG problem setup complete.")

print("Starting with assembly of the global matrices...")

# data arrays to save element matrices and vectors
data_M_full = [np.zeros((mesh_1d.n_dofs_element, mesh_1d.n_dofs_element)) for _ in range(mesh_1d.n_elements_total)]
data_K_full = [np.zeros((mesh_1d.n_dofs_element, mesh_1d.n_dofs_element)) for _ in range(mesh_1d.n_elements_total)]
data_M = [np.zeros((mesh_1d.n_dofs_element, mesh_1d.n_dofs_element)) for _ in range(mesh_1d.n_elements_total)]
data_K = [np.zeros((mesh_1d.n_dofs_element, mesh_1d.n_dofs_element)) for _ in range(mesh_1d.n_elements_total)]
data_proj = [np.zeros((mesh_1d.n_dofs_element, mesh_1d.n_dofs_element)) for _ in range(mesh_1d.n_elements_total)]
data_f_ex_v = [np.zeros(mesh_1d.n_dofs_element) for _ in range(mesh_1d.n_elements_total)]
data_ic_0_v = [np.zeros(mesh_1d.n_dofs_element) for _ in range(mesh_1d.n_elements_total)]
data_f_ex_sig = [np.zeros(mesh_1d.n_dofs_element) for _ in range(mesh_1d.n_elements_total)]
data_ic_0_sig = [np.zeros(mesh_1d.n_dofs_element) for _ in range(mesh_1d.n_elements_total)]


for element_indices in tqdm(itertools.product(*[range(n) for n in np.flip(n_elements)]), 
                     total=mesh_1d.n_elements_total, desc="Processing elements"):
    element_indices = element_indices[::-1]
    tot_index = mesh_1d.get_element_index_total(element_indices)
    tqdm.write(f"Current element indices: {element_indices}, total index: {tot_index}")

    partitions = spacetree.generate_spacetree(element_indices, depth, n_seed, mesh_1d, domain)
    cell_is_cut = len(partitions) > 1

    for partition in partitions:
        (r0, r1) = partition[0]
        r_values_rule1 = [xi * (r1 - r0) / 2 + (r1 + r0) / 2 for xi in int_points_rule1[0]]
        r_values_rule2 = [xi * (r1 - r0) / 2 + (r1 + r0) / 2 for xi in int_points_rule2[0]]

        # Evaluate shape functions and their derivatives at integration points
        # rule 1 for mass: just 0th derivative neeed
        # rule 2 for everything else: 0th and 1st derivative needed
        shapes_r_rule1_0 = [np.array([shape(r) for shape in shapes_0[0]]) for r in r_values_rule1]
        shapes_r_rule2_0 = [np.array([shape(r) for shape in shapes_0[0]]) for r in r_values_rule2]
        shapes_r_rule2_1 = [np.array([shape(r) for shape in shapes_1[0]]) for r in r_values_rule2]
    
        for igauss, (r, w0) in enumerate(zip(r_values_rule1, int_weights_rule1[0])):
            # scale weight for physical element
            weight = w0 * mesh_1d.dx / 2 * (r1 - r0) / 2
            # Add contributions
            N = shapes_r_rule1_0[igauss]
            data_M_full[tot_index] += weight * np.outer(N, N)
            if domain(mesh_1d.map_to_physical(element_indices, [r])) < 0.5:
                weight *= alpha
            data_M[tot_index] += weight * np.outer(N, N)
        
        for igauss, (r, w0) in enumerate(zip(r_values_rule2, int_weights_rule2[0])):
            # scale weight for physical element
            weight = w0 * mesh_1d.dx / 2 * (r1 - r0) / 2
            # Add contributions
            N0 = shapes_r_rule2_0[igauss]
            N1 = shapes_r_rule2_1[igauss] * (2.0 / mesh_1d.dx[0])  # chain rule for derivative
            data_K_full[tot_index] += weight * np.outer(N1, N0)
            if domain(mesh_1d.map_to_physical(element_indices, [r])) < 0.5:
                weight *= alpha
            data_K[tot_index] += weight * np.outer(N1, N0)
            data_proj[tot_index] += weight * np.outer(N0, N0)
            x_phys = mesh_1d.map_to_physical(element_indices, [r])[0]
            data_f_ex_v[tot_index] += weight * f_ex_func_v(x_phys) * N0
            data_ic_0_v[tot_index] += weight * ic_0_func_v(x_phys) * N0
            data_f_ex_sig[tot_index] += weight * f_ex_func_sig(x_phys) * N0
            data_ic_0_sig[tot_index] += weight * ic_0_func_sig(x_phys) * N0
    
    data_M[tot_index] = np.linalg.inv(data_M[tot_index])
    data_M_full[tot_index] = np.linalg.inv(data_M_full[tot_index])

print("Completed assembly of the global matrices...")

print("Time integration (" + str(n_steps) + " time steps) ... ")

# Allocate Solution vector; displacement u is integrated along the time integraiton
# Q size: time steps x elements x nodes x fields (v, sigma)
U = np.zeros((n_steps+1, mesh_1d.n_elements_total, mesh_1d.n_dofs_element))
Q = np.zeros((n_steps+1, mesh_1d.n_elements_total, mesh_1d.n_dofs_element, 2))

# Set initial conditions
for e in range(mesh_1d.n_elements_total):
    Q[0, e, :, 0] = np.linalg.solve(data_proj[e], data_ic_0_v[e])
    Q[0, e, :, 1] = np.linalg.solve(data_proj[e], data_ic_0_sig[e])

# Time integration
for n in range(n_steps):
    # stage 1: Euler step
    k1 = fluxes.compute_rhs(Q[n], mesh_1d, An, absAn, An_fict, absAn_fict, data_M, data_K, data_M_full, data_K_full, domain, mat, alpha)
    k1_U = Q[n,:,:,0]

    if time_scheme == "Euler":
        Q[n + 1] = Q[n] + dt * k1
        U[n + 1] = U[n] + dt * k1_U

    elif time_scheme == "RK2":
        Q1 = Q[n] + dt * k1
        # stage 2:
        k2 = fluxes.compute_rhs(Q1, mesh_1d, An, absAn, An_fict, absAn_fict, data_M, data_K, data_M_full, data_K_full, domain, mat, alpha)
        k2_U = Q1[:,:,0]
        # SSPRK2 update:
        Q[n+1] = Q[n] + 0.5 * dt * (k1 + k2)
        U[n+1] = U[n] + 0.5 * dt * (k1_U + k2_U)

    else:
        raise NotImplementedError

    # Track maximum velocity to see if integration is stable
    print(f"Time step {n} from {n_steps}: Maximum velocity {np.amax(Q[n+1, :, :, 0])}")

# animation of all time steps
print("Start animation of the results.")
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(origin[0], origin[0] + length[0])
ax.set_ylim(-.2, 2.2)
ax.set_xlabel('x')
ax.set_ylabel('u(x,t)')
ax.set_title('1D SEM Wave Propagation')
def init():
    line.set_data([], [])
    return line,
x = np.array([mesh_1d.map_to_physical_dim(0, e, gll_nodes[0]) for e in range(n_elements[0])]).ravel()
def animate(i):
    y = Q[i, :, :, 0].ravel()
    line.set_data(x, y)
    ax.vlines(x_bounds, -0.2, 2.2, colors='r', linestyles='dashed', label='Immersed Boundary')
    ax.set_title(f'1D SEM Wave Propagation at t={i*dt:.3f}s')
    return line, ax
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=range(0, n_steps+1, 100), interval=1, blit=True)
plt.show()