import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import itertools
from tqdm import tqdm

import material
import mesh
import integration
import ansatz
import functions
import geometries
import spacetree


# Spatial dimension
dim = 1
print("Setting up 1D SEM problem...")

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
x_bounds = (0.0, 2/3)
# domain = geometries.immersed_1d_bar(x_bounds=x_bounds)
domain = geometries.all_physical()

# stabilization parameters for cut cells and parameter for spacetree
alpha = 1e-2
depth = np.max(polynomial_degree) + 1
n_seed = 10

# material 
density = 1.0
wave_speed = 1.0
print(f"Material properties - Density: {density}, Wave Speed: {wave_speed}")
mat = material.acousticMaterial(density=density, wave_speed=wave_speed)

# time
T = 2 * length[0] / wave_speed
n_steps = 1000
print(f"Total simulation time: {T}, Number of time steps: {n_steps}")
dt = T / n_steps

# initial conditions
frequency = 1.0
x0 = 0.0
sigma = wave_speed / frequency / 20
print(f"Initial condition - Gaussian pulse centered at {x0} with sigma {sigma}")
ic_func = functions.gaussian_pulse_1d(x0=x0, sigma=sigma, wave_speed=wave_speed)
def ic_0_func(x):
    return ic_func(x, 0.0)
def ic_1_func(x):
    return ic_func(x, dt)

# external force
def f_ex_func(x): 
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
    int_p, int_w = integration.get_gll_integration(polynomial_degree[i_d] + 1)
    int_points_rule1.append(int_p)
    int_weights_rule1.append(int_w)
    int_p, int_w = np.polynomial.legendre.leggauss(polynomial_degree[i_d] + 1)
    int_points_rule2.append(int_p)
    int_weights_rule2.append(int_w)

print("1D SEM problem setup complete.")

print("Starting with assembly of the global matrices...")

# data arrays global matrices in COO format
n_entries_assembly = mesh_1d.n_elements_total * mesh_1d.n_dofs_element ** 2
row = np.zeros(n_entries_assembly, dtype=np.uint)
col = np.zeros(n_entries_assembly, dtype=np.uint)
data_M = np.zeros(n_entries_assembly)
data_K = np.zeros(n_entries_assembly)
data_proj = np.zeros(n_entries_assembly)

# data arrays for global external force and initial conditions
f_ex = np.zeros(mesh_1d.n_dofs_global)
ic_0 = np.zeros(mesh_1d.n_dofs_global)
ic_1 = np.zeros(mesh_1d.n_dofs_global)

for element_indices in tqdm(itertools.product(*[range(n) for n in np.flip(n_elements)]), 
                     total=mesh_1d.n_elements_total, desc="Processing elements"):
    element_indices = element_indices[::-1]
    tqdm.write(f"Current element indices: {element_indices}")

    # Local element matrices 
    proje = np.zeros((mesh_1d.n_dofs_element, mesh_1d.n_dofs_element))
    Me = np.zeros((mesh_1d.n_dofs_element, mesh_1d.n_dofs_element))
    Ke = np.zeros((mesh_1d.n_dofs_element, mesh_1d.n_dofs_element))

    # Local element vectors
    f_exe = np.zeros((mesh_1d.n_dofs_element,))
    ic_0e = np.zeros((mesh_1d.n_dofs_element,))
    ic_m1e = np.zeros((mesh_1d.n_dofs_element,))

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
            if domain(mesh_1d.map_to_physical(element_indices, [r])) < 0.5:
                weight *= alpha
            # Add contributions
            N = shapes_r_rule1_0[igauss]
            Me += weight * mat.density * np.outer(N, N)
        
        for igauss, (r, w0) in enumerate(zip(r_values_rule2, int_weights_rule2[0])):
            # scale weight for physical element
            weight = w0 * mesh_1d.dx / 2 * (r1 - r0) / 2
            if domain(mesh_1d.map_to_physical(element_indices, [r])) < 0.5:
                weight *= alpha
            # Add contributions
            N0 = shapes_r_rule2_0[igauss]
            N1 = shapes_r_rule2_1[igauss] * (2.0 / mesh_1d.dx[0])  # chain rule for derivative
            Ke += weight * mat.density / mat.wave_speed ** 2 * np.outer(N1, N1)
            proje += weight * np.outer(N0, N0)
            x_phys = mesh_1d.map_to_physical(element_indices, [r])[0]
            f_exe += weight * f_ex_func(x_phys) * N0
            ic_0e += weight * ic_0_func(x_phys) * N0
            ic_m1e += weight * ic_1_func(x_phys) * N0
            
    
    # get the location map for the current element
    lm = mesh_1d.get_location_map(element_indices)
    # Repeat locationMaps as colums/rows linearize the result to obtain matrix entry coordinates
    element_slice = mesh_1d.get_element_slice(element_indices)

    row[element_slice] = np.broadcast_to(lm, (mesh_1d.n_dofs_element, mesh_1d.n_dofs_element)).ravel()
    col[element_slice] = np.broadcast_to(lm, (mesh_1d.n_dofs_element, mesh_1d.n_dofs_element)).T.ravel()

    data_M[element_slice] = Me.ravel()
    data_K[element_slice] = Ke.ravel()
    data_proj[element_slice] = proje.ravel()
    f_ex[lm] += f_exe
    ic_0[lm] += ic_0e
    ic_1[lm] += ic_m1e

# Create coordinate matrix and convert to compressed sparse column format
M = scipy.sparse.coo_matrix((data_M, (row, col)), shape=(mesh_1d.n_dofs_global, mesh_1d.n_dofs_global)).tocsc()
K = scipy.sparse.coo_matrix((data_K, (row, col)), shape=(mesh_1d.n_dofs_global, mesh_1d.n_dofs_global)).tocsc()
proj = scipy.sparse.coo_matrix((data_proj, (row, col)), shape=(mesh_1d.n_dofs_global, mesh_1d.n_dofs_global)).tocsc()

# for boundary-conforming SEM we can take the diagonal of the mass matrix and invert it directly
print("Average bandwidth before filtering: " + "%.2f" % (M.nnz / M.shape[0]))
M.data[np.abs(M.data) < 1e-16 * scipy.sparse.linalg.norm(M)] = 0.0
M.eliminate_zeros()
print("Average bandwidth after filtering: " + "%.2f" % (M.nnz / M.shape[0]))

print("Completed assembly of the global matrices...")

print("Time integration (" + str(n_steps) + " time steps) ... ")

# Sparse LU factorization of and time integration
factorized_M = scipy.sparse.linalg.splu(M)
factorized_proj = scipy.sparse.linalg.splu(proj)

u = np.zeros((n_steps + 1, mesh_1d.n_dofs_global))
u[0] = factorized_proj.solve(ic_0)
u[1] = factorized_proj.solve(ic_1)

for i_step in tqdm(range(1, n_steps), desc="Time stepping"):
    # compute internal + external forces
    rhs = dt ** 2 *(- K.dot(u[i_step]) + f_ex * f_t(i_step * dt))
    u[i_step + 1] = 2.0 * u[i_step] - u[i_step - 1] + factorized_M.solve(rhs)
    
print("Time integration complete.")

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
x = np.empty(mesh_1d.n_dofs_global)
for i_el in range(n_elements[0]):
    for i_p, point in enumerate(gll_nodes[0]):
        x[i_el * (polynomial_degree[0]) + i_p] = mesh_1d.map_to_physical([i_el], [point])[0]
def animate(i):
    y = u[i]
    line.set_data(x, y)
    ax.vlines(x_bounds, -0.2, 2.2, colors='r', linestyles='dashed', label='Immersed Boundary')
    ax.set_title(f'1D SEM Wave Propagation at t={i*dt:.3f}s')
    return line, ax
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=range(0, n_steps+1, 4), interval=1, blit=True)
plt.show()